package main

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"sort"
	"strings"
)

// --- the machines the organisers rent ----------------------------------------------------------
//
// One of the organisers' own keys (STET_ANCHOR_KEYS) vouches for a rented machine's key by
// signing a grant (WorkerGrant), and the server keeps it. So renting a machine needs no
// change to the server's configuration: stet-cloud makes a key, vouches for it with the
// organisers' key it is run with, and the machine is one of the organisers' from then on.
// Taking the grant back, or the organisers' key that made it out of the configuration,
// makes it nobody's.

// maxWorkers bounds the grants kept: far more machines than the organisers would rent.
const maxWorkers = 256

// WorkerView is a grant as the organisers are shown it.
type WorkerView struct {
	Key      string `json:"key"`
	Handle   string `json:"handle,omitempty"` // its account, once it has one
	By       string `json:"by"`               // the organisers' account that vouched for it
	At       int64  `json:"at"`
	Counts   bool   `json:"counts"` // its signature checks out and its maker is one of the organisers' keys now
	LastSeen int64  `json:"last_seen,omitempty"`
	Busy     bool   `json:"busy,omitempty"` // it holds a job
}

func (a *App) apiWorkers(w http.ResponseWriter, r *http.Request) {
	if _, _, _, err := a.signedBy(w, r, 1<<10, true, organisers); err != nil {
		writeErr(w, err)
		return
	}
	out := []WorkerView{}
	a.st.view(func(s *State) {
		busy := map[int64]bool{}
		for _, c := range s.Claims {
			if c.live(now()) {
				busy[c.AccountID] = true
			}
		}
		for _, g := range s.Workers {
			v := WorkerView{Key: g.Key, By: g.By, At: g.At, Counts: g.ok && a.cfg.AnchorKeys[g.By]}
			if by := s.byPubkey[g.By]; by != nil {
				v.By = by.Handle
			}
			if acct := s.byPubkey[g.Key]; acct != nil {
				v.Handle, v.LastSeen, v.Busy = acct.Handle, acct.LastSeen, busy[acct.ID]
			}
			out = append(out, v)
		}
	})
	sort.Slice(out, func(i, j int) bool { return out[i].Handle+out[i].Key < out[j].Handle+out[j].Key })
	writeJSON(w, 200, map[string]any{"workers": out})
}

// apiWorkerGrant keeps an organiser's grant for a machine they rent. The request is signed by
// the organiser, as every request is; the grant is signed too, and kept with its signature,
// so that what is in the database proves itself each time it is read.
func (a *App) apiWorkerGrant(w http.ResponseWriter, r *http.Request) {
	acct, _, body, err := a.signedBy(w, r, 1<<12, true, organisers)
	if err != nil {
		writeErr(w, err)
		return
	}
	var req struct {
		Key string `json:"key"`
		At  int64  `json:"at"`
		Sig string `json:"sig"`
	}
	if json.Unmarshal(body, &req) != nil {
		writeErr(w, fail(400, "malformed request"))
		return
	}
	g := &WorkerGrant{Key: strings.ToLower(req.Key), By: acct.Pubkey, At: req.At, Sig: strings.ToLower(req.Sig)}
	switch {
	case !sha256Re.MatchString(g.Key):
		writeErr(w, fail(400, "that is not a public key"))
		return
	case a.cfg.AnchorKeys[g.Key]:
		writeErr(w, fail(400, "that key is one of the organisers' own already"))
		return
	case g.At < now()-3600 || g.At > now()+3600:
		writeErr(w, fail(400, "the grant is dated more than an hour from now"))
		return
	case !g.check():
		writeErr(w, fail(400, "the grant is not signed by the organisers' key that sent it"))
		return
	}
	var handle string
	err = a.st.update(r.Context(), func(tx *Tx) error {
		s := tx.s
		if s.Workers[g.Key] == nil && len(s.Workers) >= maxWorkers {
			return fail(409, "the organisers already vouch for %d machines: take some back first", maxWorkers)
		}
		s.Workers[g.Key] = g
		tx.put(workerKey(g.Key), g)
		s.markOrganisers(a.cfg.AnchorKeys)
		if other := s.byPubkey[g.Key]; other != nil {
			handle = other.Handle
		}
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 201, map[string]any{"key": g.Key, "handle": handle, "by": acct.Handle, "at": g.At})
}

// apiWorkerRevoke takes a grant back. The machine's account stays, and is nobody's machine.
func (a *App) apiWorkerRevoke(w http.ResponseWriter, r *http.Request) {
	if _, _, _, err := a.signedBy(w, r, 1<<10, true, organisers); err != nil {
		writeErr(w, err)
		return
	}
	key := strings.ToLower(r.PathValue("key"))
	err := a.st.update(r.Context(), func(tx *Tx) error {
		s := tx.s
		if s.Workers[key] == nil {
			return fail(404, "the organisers do not vouch for that key")
		}
		delete(s.Workers, key)
		tx.put(workerKey(key), nil)
		s.markOrganisers(a.cfg.AnchorKeys)
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"key": key, "revoked": true})
}

// endedReason is what a job handed back by apiWorkerRelease says.
const endedReason = "the organisers ended the machine"

// apiWorkerRelease hands back every job a rented machine holds, once the organisers have ended
// it: a machine ended a moment after it took a job would otherwise hold the job, and the round
// or an organisers' batch waiting on it, until its lease ran out. The official jobs go back
// as if the machine had handed them back itself, and so do the organisers' own students.
func (a *App) apiWorkerRelease(w http.ResponseWriter, r *http.Request) {
	op, _, _, err := a.signedBy(w, r, 1<<10, true, organisers)
	if err != nil {
		writeErr(w, err)
		return
	}
	key := strings.ToLower(r.PathValue("key"))
	var handle string
	jobs, students := 0, 0
	err = a.st.update(r.Context(), func(tx *Tx) error {
		s := tx.s
		if s.Workers[key] == nil {
			return fail(404, "the organisers do not vouch for that key")
		}
		acct := s.byPubkey[key]
		if acct == nil {
			return nil // it never asked for work, so it holds none
		}
		handle = acct.Handle
		t := now()
		acct.EndedAt = t
		tx.saveAccount(acct)
		for _, c := range s.Claims {
			if c.AccountID == acct.ID && c.live(t) {
				a.handBack(tx, c, endedReason)
				jobs++
			}
		}
		for _, b := range s.Farm {
			held := false
			for _, task := range b.Tasks {
				if task.Done == nil && task.Holder == acct.ID && task.Until > t {
					task.Holder, task.Until = 0, 0
					held = true
					students++
				}
			}
			if held {
				tx.saveFarmBatch(b)
			}
		}
		if jobs > 0 {
			a.settleReady(tx) // a round may have been waiting for these jobs alone
		}
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	if jobs+students > 0 {
		slog.Info("jobs handed back for a machine the organisers ended", "machine", handle, "by", op.Handle, "jobs", jobs, "students", students)
	}
	writeJSON(w, 200, map[string]any{"key": key, "handle": handle, "jobs": jobs, "students": students})
}
