# A machine the organisers rent from RunPod: it installs stet, grades for the leaderboard
# until nothing is left for it, and then ends its own pod. stet-runpod starts it as the
# pod's only process (bash -c), with these in its environment:
#
#   STET_WORKER_KEY     the machine's stet key file, base64
#   STET_API            the Stet API
#   STET_INSTALL        the site's installer
#   STET_IDLE_MINUTES   how long it waits with nothing to do before it stops
#   PUBLIC_KEY          optional (--ssh): a key that may log in, to look around
#   STET_WORKER_HOME    where it keeps its files: /root, but for the tests
#
# RunPod gives every pod RUNPOD_POD_ID, and RUNPOD_API_KEY: a key that reaches only this
# pod, which is what it ends itself with. A container whose process exits is started again
# and billed as before, so this never just exits: it ends the pod, and if RunPod will not
# have that, it waits for stet-runpod to end it from outside.
set -u
home=${STET_WORKER_HOME:-/root}
export HOME="$home" DEBIAN_FRONTEND=noninteractive PATH="$home/.local/bin:$PATH"
done_mark=$home/.stet-worker-done

say() { printf '[stet-worker %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

end_pod() {
  touch "$done_mark"
  say "ending this pod: $1"
  if ! command -v curl >/dev/null 2>&1; then
    say "there is no curl to end it with; stet-runpod ends it from outside"
    exec sleep infinity
  fi
  tries=0
  while :; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE \
      -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID:-none}" || true)
    case "$code" in
      204|404) say "RunPod has it; this pod is ending"; exec sleep infinity ;;
    esac
    tries=$((tries + 1))
    say "RunPod did not end it (HTTP $code); trying again, and stet-runpod ends it from outside meanwhile"
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

# In the background, so that being told to stop (the pod ended from outside) reaches stet,
# which hands back the job it holds.
stet verify --exit-when-idle "${STET_IDLE_MINUTES:-10}" &
child=$!
trap 'say "told to stop"; kill -TERM "$child" 2>/dev/null; wait "$child"; touch "$done_mark"; exit 0' TERM INT
wait "$child"
end_pod "stet verify stopped (exit $?)"
