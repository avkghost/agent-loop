#!/bin/sh
# RME-1.1 egress initialization. Runs as root BEFORE any model-controlled process. Fails closed.
# Policy: all outbound traffic dropped (IPv4 and IPv6) except loopback, established flows, and TCP 1234 to the
# address that host.docker.internal resolves to inside this container.
set -eu
die() { echo "EGRESS_INIT_FAILED: $*" >&2; exit 111; }

command -v iptables >/dev/null 2>&1 || die "iptables missing"
command -v ip6tables >/dev/null 2>&1 || die "ip6tables missing"
command -v setpriv >/dev/null 2>&1 || die "setpriv missing"
[ "$(id -u)" = "0" ] || die "must start as root to install rules"

IP=$(getent ahostsv4 host.docker.internal | awk 'NR==1{print $1}') || die "cannot resolve host.docker.internal"
[ -n "$IP" ] || die "empty resolution"
echo "$IP" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' || die "unexpected address: $IP"
PORT=1234

iptables -F OUTPUT || die "iptables flush"
iptables -P OUTPUT DROP || die "iptables policy"
iptables -A OUTPUT -o lo -j ACCEPT || die "iptables loopback"
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT || die "iptables established"
iptables -A OUTPUT -p tcp -d "$IP" --dport "$PORT" -j ACCEPT || die "iptables allow LM Studio"
iptables -A OUTPUT -j DROP || die "iptables final drop"

ip6tables -F OUTPUT || die "ip6tables flush"
ip6tables -P OUTPUT DROP || die "ip6tables policy"
ip6tables -A OUTPUT -o lo -j ACCEPT || die "ip6tables loopback"
ip6tables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT || die "ip6tables established"
ip6tables -A OUTPUT -j DROP || die "ip6tables final drop"

# Verify what was installed, rather than trusting the commands above
iptables -S OUTPUT | grep -qx -- "-P OUTPUT DROP" || die "IPv4 policy not verified"
iptables -S OUTPUT | grep -q -- "-d $IP/32 -p tcp -m tcp --dport $PORT -j ACCEPT" || die "LM Studio rule not verified"
ip6tables -S OUTPUT | grep -qx -- "-P OUTPUT DROP" || die "IPv6 policy not verified"

echo "EGRESS_POLICY: drop all except tcp $IP:$PORT (and loopback, established)"
export RME_LLM_URL="http://$IP:$PORT/v1/chat/completions"

# Drop to the agent user with no capabilities and no way to regain them. Afterwards NET_ADMIN is unusable.
exec setpriv --reuid=10001 --regid=10001 --clear-groups --inh-caps=-all --bounding-set=-all --no-new-privs -- "$@"
