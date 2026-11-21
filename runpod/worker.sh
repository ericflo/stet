# A machine the organisers rent: it installs stet, grades for the leaderboard until nothing
# is left for it, and then ends itself where its market lets it. stet-cloud starts it as the
# machine's only process (bash -c), with these in its environment:
#
#   STET_WORKER_KEY     the machine's stet key file, base64
#   STET_API            the Stet API
#   STET_INSTALL        the site's installer
#   STET_IDLE_MINUTES   how long it waits with nothing to do before it stops
#   STET_WORKER_EDITS   1 to run curators too, in the container sandbox (stet's cage.py)
#   PUBLIC_KEY          optional (--ssh): a key that may log in, to look around
#   STET_WORKER_HOME    where it keeps its files: /root, but for the tests
#   STET_MARKET         where it is rented: runpod (when not set) or vast; elsewhere it
#                       cannot end itself, and waits for stet-cloud to end it
#   STET_PROBE          optional (stet-cloud probe): tools/cage_check.py, gzipped and base64'd,
#                       or in pieces STET_PROBE_0, STET_PROBE_1, ... where a market's values
#                       are short. It then takes no work: it says what this machine is, what
#                       stet doctor says, and whether the container sandbox holds against a
#                       curator that wants out, and waits to be ended.
#
# RunPod gives every pod RUNPOD_POD_ID, and RUNPOD_API_KEY: a key that reaches only this
# pod. Vast gives every instance CONTAINER_ID and CONTAINER_API_KEY, the same. That is what
# it ends itself with. A container whose process exits is started again and billed as
# before, or its disk is billed until it is destroyed, so this never just exits: it ends
# itself, and if the market will not have that, it waits for stet-cloud to end it.
set -u
home=${STET_WORKER_HOME:-/root}
opt=${STET_WORKER_OPT:-/opt/stet}
# stet, its Python and the season's libraries go under /opt, where the curator's user can
# read them; stet's key, cache and temporary files stay under /root, where it cannot.
export HOME="$home" DEBIAN_FRONTEND=noninteractive PATH="$opt/bin:$home/.local/bin:$PATH" \
  UV_TOOL_DIR="$opt/tools" UV_PYTHON_INSTALL_DIR="$opt/python" UV_TOOL_BIN_DIR="$opt/bin" TMPDIR="$home/.cache/stet-tmp"
mkdir -p "$TMPDIR" && chmod 700 "$home" "$TMPDIR" 2>/dev/null
done_mark=$home/.stet-worker-done

say() { printf '[stet-worker %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

end_pod() {
  touch "$done_mark"
  market=${STET_MARKET:-runpod}
  case "$market" in
    runpod) name=RunPod what=pod url="https://api.runpod.io/v2/pods/${RUNPOD_POD_ID:-none}" key=${RUNPOD_API_KEY:-} ok="204|404" ;;
    vast) name=Vast what=instance url="https://console.vast.ai/api/v0/instances/${CONTAINER_ID:-none}/" key=${CONTAINER_API_KEY:-} ok="200|404" ;;
    *) name=$market what=machine url="" key="" ok="" ;;
  esac
  say "ending this $what: $1"
  if [ -z "$url" ]; then
    say "$name gives it no key to end itself with; stet-cloud ends it from outside"
    exec sleep infinity
  fi
  if ! command -v curl >/dev/null 2>&1 || [ -z "$key" ]; then
    say "there is no curl, or no key, to end it with; stet-cloud ends it from outside"
    exec sleep infinity
  fi
  tries=0
  while :; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE -H "Authorization: Bearer $key" "$url" || true)
    if printf '%s' "$code" | grep -qxE "$ok"; then
      say "$name has it; this $what is ending"
      exec sleep infinity
    fi
    tries=$((tries + 1))
    say "$name did not end it (HTTP $code); trying again, and stet-cloud ends it from outside meanwhile"
    if [ "$tries" -lt 10 ]; then sleep 60; else sleep 600; fi
  done
}

# Started again after it had finished (RunPod restarts a container that stops): finish.
[ -e "$done_mark" ] && end_pod "it was started again after it had finished"

if ! command -v curl >/dev/null 2>&1 || [ -n "${PUBLIC_KEY:-}" ]; then
  pkgs="curl ca-certificates"
  [ -n "${PUBLIC_KEY:-}" ] && pkgs="$pkgs openssh-server"
  for i in 1 2 3; do
    apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq --no-install-recommends $pkgs >/dev/null 2>&1 && break
    say "could not install $pkgs; trying again"
    sleep 20
  done
fi
command -v curl >/dev/null 2>&1 || end_pod "curl could not be installed"

# Salad does not say its hosts' drivers. One older than the season's PyTorch needs (580, for
# CUDA 13) asks Salad for another machine, which starts this again there.
if [ "${STET_MARKET:-}" = "salad" ] && command -v nvidia-smi >/dev/null 2>&1; then
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
  if [ -n "$driver" ] && [ "${driver%%.*}" -lt 580 ] 2>/dev/null; then
    say "driver $driver is older than 580: asking Salad for another machine"
    curl -s -X POST http://169.254.169.254/v1/reallocate -H 'Content-Type: application/json' -H 'Metadata: true' \
      -d "{\"reason\": \"driver $driver is older than 580\"}" >/dev/null || true
    exec sleep infinity
  fi
fi

if [ -n "${PUBLIC_KEY:-}" ] && [ -x /usr/sbin/sshd ]; then
  mkdir -p "$home/.ssh" /run/sshd && chmod 700 "$home/.ssh"
  printf '%s\n' "$PUBLIC_KEY" > "$home/.ssh/authorized_keys" && chmod 600 "$home/.ssh/authorized_keys"
  /usr/sbin/sshd && say "ssh is open to the key it was given"
fi

# The key, readable by nobody else, and out of the environment that stet runs in.
umask 077
mkdir -p "$home/.config/stet"
if ! printf '%s' "${STET_WORKER_KEY:-}" | base64 -d > "$home/.config/stet/key.json" 2>/dev/null || [ ! -s "$home/.config/stet/key.json" ]; then
  end_pod "it was not given a key"
fi
unset STET_WORKER_KEY
umask 022

if ! command -v stet >/dev/null 2>&1; then
  say "installing stet from ${STET_INSTALL}"
  for i in 1 2 3; do
    if command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; then
      curl -fsSL "$STET_INSTALL" | sh && break
    fi
    say "the install did not work; trying again"
    sleep 30
  done
fi
command -v stet >/dev/null 2>&1 || end_pod "stet could not be installed"
say "$(stet --version)"

# A probe: what this machine is, and whether stet's container sandbox holds here. Every line
# it says starts "[probe]", and is kept in $home/probe.txt too.
if [ -z "${STET_PROBE:-}" ]; then
  for i in 0 1 2 3 4 5 6 7 8 9; do v="STET_PROBE_$i"; STET_PROBE="${STET_PROBE:-}${!v:-}"; done
fi
if [ -n "${STET_PROBE:-}" ]; then
  probe() { sed -u 's/^/[probe] /' | tee -a "$home/probe.txt"; }
  {
    echo "machine: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | head -n 1)"
    echo "kernel: $(uname -r); cpus: $(nproc); memory: $(free -g | awk '/^Mem/ {print $2}') GB; disk free: $(df -h "$home" | awk 'NR==2 {print $4}')"
    echo "landlock: $(cat /sys/kernel/security/lsm 2>/dev/null || echo unknown); cgroup: $(stat -fc %T /sys/fs/cgroup 2>/dev/null)"
  } | probe
  stet doctor 2>&1 | sed 's/^/doctor: /' | probe
  printf '%s' "$STET_PROBE" | base64 -d | gunzip > "$home/cage_check.py" 2>/dev/null
  py=$(head -n 1 "$(command -v stet)" | sed 's/^#!//')
  if [ -s "$home/cage_check.py" ] && [ -x "$py" ]; then
    (cd "$home" && "$py" "$home/cage_check.py" 2>&1) | sed 's/^/cage: /' | probe
    echo "done: cage_check exit ${PIPESTATUS[0]}" | probe
  else
    echo "done: no cage_check to run (python: $py)" | probe
  fi
  say "probe finished; waiting to be ended"
  exec sleep infinity
fi

# In the background, so that being told to stop (the pod ended from outside) reaches stet,
# which hands back the job it holds.
verify() {
  stet verify "$@" --exit-when-idle "${STET_IDLE_MINUTES:-10}" &
  child=$!
  wait "$child"
}
trap 'say "told to stop"; kill -TERM "$child" 2>/dev/null; wait "$child"; touch "$done_mark"; exit 0' TERM INT
if [ "${STET_WORKER_EDITS:-0}" = "1" ]; then
  verify --edits --sandbox container
  rc=$?
  # It stopped for want of work, or could not run curators here at all: the sandbox would
  # not hold, the machine is under the floor. Then it grades, which needs neither.
  if [ "$rc" -ne 0 ]; then
    say "stet verify --edits stopped (exit $rc); grading only"
    verify
    rc=$?
  fi
else
  verify
  rc=$?
fi
end_pod "stet verify stopped (exit $rc)"
