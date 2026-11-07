# Run Folivox in Docker

For a home server or NAS: the web studio runs in a container, you open it from any computer on your network, and the
finished audiobooks land in a folder you choose (for example your Audiobookshelf library). It works from Portainer and
Dockhand as a stack, or with plain `docker compose`. Nothing has to be typed in a terminal on the server.

> **Mac:** Docker on macOS cannot use the Apple GPU (no Metal in containers), so the container would run on the CPU,
> which is slow. On a Mac use the normal [installer](install.md). The same goes for a PC with a GPU you do not want to
> pass into Docker: the installer is simpler there.

> **Tested so far:** the CPU image on Linux arm64 (build, start, login, health check, file ownership, stop). The
> `cuda` and `cuda128` images, NVIDIA GPUs, Docker Desktop on Windows with WSL 2, Portainer and Dockhand stacks, and
> the amd64 build have **not been tested** yet; the instructions for them below are what should work, not what was seen
> working. Please report what you find.

## Quick start

1. Pick a password for the studio: at least 4 characters (longer is safer, for example the output of `openssl rand -hex 12`).
   You can also run without a password on a trusted home network: see [Without a password](#without-a-password-trusted-network-only).
2. Create a stack from [docker-compose.yml](../docker-compose.yml) (CPU) or
   [docker-compose.nvidia.yml](../docker-compose.nvidia.yml) (NVIDIA GPU). In Portainer: *Stacks → Add stack → Web
   editor*, paste the file, and under *Environment variables* add `FOLIVOX_UI_TOKEN` set to your password (and the optional ones below). Without a password, leave the token empty and set
   `FOLIVOX_UI_AUTH` to `off` instead.
3. Deploy. Open `http://<server>:8765/`, enter the password (if you set one), and use the studio as on a desktop.

If you set neither a token nor `FOLIVOX_UI_AUTH=off`, the container keeps running but shows only a notice on the port
("Folivox needs one setting") and reports *unhealthy*; set one of them and redeploy.

With plain Docker Compose:

```bash
export FOLIVOX_UI_TOKEN=$(openssl rand -hex 12)   # keep it: this is the password
echo "$FOLIVOX_UI_TOKEN"
docker compose up -d
```

The first render downloads the models (about 7.6 GB: the voice and the speech-recognition check) with a progress bar in
the studio. To fetch them at start instead, set `FOLIVOX_PRELOAD_MODELS=1`: the studio then starts only after the
download has finished (progress is in the container log; the container shows as *starting* meanwhile). They are
stored in the `folivox-data` volume, so updating or recreating the container does not download them again.

## Images

| Tag | For | Platforms |
|---|---|---|
| `ghcr.io/vatavale/folivox:cpu` (also `latest`) | No GPU | amd64, arm64 |
| `ghcr.io/vatavale/folivox:cuda` | NVIDIA GPU (PyTorch 2.5, CUDA 12.4) | amd64 |
| `ghcr.io/vatavale/folivox:cuda128` | NVIDIA RTX 50 series (PyTorch 2.7.1, CUDA 12.8) | amd64 |

Every image is also tagged with the version (`cpu-0.1.0.N`). The images contain the program and ffmpeg, not the models.
To build one yourself: `docker build -f docker/Dockerfile --build-arg VARIANT=cpu -t folivox:cpu .`
(`VARIANT` is `cpu`, `cuda` or `cuda128`).

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `FOLIVOX_UI_TOKEN` | required | The studio password (4+ characters; longer is safer). The studio refuses to start on the network without it. |
| `FOLIVOX_UI_TOKEN_FILE` | | Path of a file holding the token (Docker/Portainer secret), instead of the variable. |
| `PUID`, `PGID` | 1000 | The user and group that own the files the container writes (`id -u`, `id -g` on the host). |
| `FOLIVOX_PORT` (compose) | 8765 | Port on the host. |
| `FOLIVOX_AUDIOBOOKS` (compose) | volume `folivox-audiobooks` | Absolute host folder for finished audiobooks (for example `/srv/audiobooks`). Leave it unset to use a Docker volume. |
| `FOLIVOX_UI_ALLOWED_HOSTS` | any | Comma-separated host names the studio answers to (see Security). |
| `FOLIVOX_UI_AUTH` | token | `off`: run without a password, for a trusted home network only (see "Without a password"). |
| `FOLIVOX_UI_ALLOWED_NETS` | private ranges | With `FOLIVOX_UI_AUTH=off`: the client networks that may connect (CIDR, comma-separated). |
| `FOLIVOX_PRELOAD_MODELS` | 0 | `1`: download the models when the container starts. |
| `FOLIVOX_RUNTIME` | auto | `auto` (PyTorch for now), `pytorch` or `native`. It wins over *Inference runtime* in the studio and over `config.toml` (which lives in the `/data` volume, `/data/config/folivox/config.toml`), so this is the way to set it when the container starts. The native engine cannot use the GPU in Docker Desktop on Windows (see "NVIDIA GPU"); use `pytorch` there. |
| `FOLIVOX_TAG` (compose) | `cpu` / `cuda` | Image tag. |
| `TZ` | UTC | Time zone of the log. |

API keys for cloud engines: the container has no OS keychain, so keys typed into the studio cannot be saved there.
Pass them as environment variables (or as files, below) instead (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `FISH_API_KEY`, `CARTESIA_API_KEY`;
see [cloud-engines.md](cloud-engines.md)); the compose files have commented lines for them.

### Docker secrets (`*_FILE`)

Every API key variable also has a `_FILE` form: `OPENAI_API_KEY_FILE=/run/secrets/openai_api_key` reads the key from
that file (surrounding whitespace and the final newline are cut). The same works for the studio password
(`FOLIVOX_UI_TOKEN_FILE`). If both `OPENAI_API_KEY` and `OPENAI_API_KEY_FILE` are set, the plain variable wins. An empty or
unreadable file is reported by name, never with its content. The studio lists such a key as coming from the environment
and a key saved in the studio does not override it (the environment wins). The compose files contain a commented `secrets:` block as a starting point;
in Portainer create the secrets in Swarm mode or mount the files into the stack folder.

## Volumes

| Mount | Holds | Keep it? |
|---|---|---|
| `/data` (`folivox-data`) | Models, voices, settings (`config.toml`), studio state, uploaded books, Hugging Face cache | Yes: re-downloading is 7.6 GB |
| `/audiobooks` (`folivox-audiobooks`, or your folder) | Finished `.m4b` / `.mp3` | Yes: point it at your audiobook folder |
| `/work` (`folivox-work`) | Journals and finished takes of books still being rendered | Yes until the book is done: it is what lets a stopped render continue |

Work files live apart from `/audiobooks` on purpose, so Audiobookshelf never scans half-made books.
Uploaded books are kept in `/data`; the studio library lists them.

## Security

The studio can read and write files in its volumes and uses the API keys you give it, so it is protected like an admin
page:

* The password is required. Listening beyond `127.0.0.1` without `FOLIVOX_UI_TOKEN` is refused at start (a normal
  desktop install is unchanged: loopback only, with an automatically generated token). The token is never written to
  the log.
* You enter it once on a login page; the browser keeps a cookie (`SameSite=Strict`, HttpOnly, 30 days; `Secure` as soon as the request came over HTTPS,
  directly or through a proxy that sends `X-Forwarded-Proto: https`). Other sites cannot make
  your browser call the studio: cross-site requests are refused, and without the cookie or token every API call
  returns 401. A wrong password costs one second per try, and ten wrong ones within a minute lock password checks for five minutes
  (a browser that is already logged in is not affected).
  The limit keeps even a short password out of reach of quick guessing (a 4-digit one would take days of
  continuous trying), but a short password is only fine on a network you trust; use a long one for anything wider.
* The connection is **plain HTTP**. Use it on your home network, **do not forward the port to the internet**. To reach
  it from outside, put a reverse proxy with HTTPS and your own login in front (Caddy, Traefik, nginx Proxy Manager,
  Cloudflare Tunnel, Tailscale) and bind the port to loopback or the proxy network (`"127.0.0.1:8765:8765"`). The
  proxy must pass the `Host` header unchanged (and `Origin` as the browser sent it). Set `FOLIVOX_UI_ALLOWED_HOSTS` to
  the proxy's name to refuse any other `Host` as a protection against DNS rebinding.
* The container runs as your `PUID`/`PGID`, never as root, so audiobooks are not owned by root. The compose files also
  drop every Linux capability except the three the start-up needs and set `no-new-privileges`.
* Anyone who has the password can make the studio read and write anywhere the container user can (the output folder is
  a setting). Mount only the folders you want it to see.

## Without a password (trusted network only)

The password is the default and the safe choice. If the studio sits on a home network where you trust every device,
you can turn it off on purpose: leave `FOLIVOX_UI_TOKEN` empty and set

```yaml
FOLIVOX_UI_AUTH: "off"
```

(in a Portainer stack: leave `FOLIVOX_UI_TOKEN` empty and add the variable `FOLIVOX_UI_AUTH` with the value `off`).
Any other value, or no value, keeps the password. Setting a token **and** `FOLIVOX_UI_AUTH=off` stops the start:
that combination would look protected without being so.

What changes: no login page, no cookie; `/healthz` and everything else open directly. The log shows one loud warning
at start. What stays: cross-site requests are still refused, and keys you saved are never sent back by the API (only
whether one is set).

What it means: **anyone who can reach the port** can start renders, spend money on your cloud API keys, and change the
output folder and anything the container user can write. Not for Wi-Fi guests, shared flats or the internet.

Guards that remain:

* **Client address.** Only clients from private and local ranges are served: `10/8`, `172.16/12`, `192.168/16`,
  `127/8`, `169.254/16`, IPv6 `fc00::/7`, `fe80::/10` and `::1`. Others get a 403 that says how to allow them. The
  address is the TCP peer; `X-Forwarded-For` is **never** trusted. To narrow or widen the list set
  `FOLIVOX_UI_ALLOWED_NETS` (it *replaces* the ranges above), for example `192.168.1.0/24` or, for Tailscale,
  `100.64.0.0/10,192.168.1.0/24`.
* **Host header** (against DNS-rebinding from a web page in your browser). Accepted: IP addresses, single-word
  names (the host name from Portainer, `nas`), and `.local`, `.lan`, `.home.arpa`, `.internal` names. A public name
  such as `studio.example.org` is refused unless you list it in `FOLIVOX_UI_ALLOWED_HOSTS` (which then decides alone).

Limits, honestly:

* **Docker Desktop, OrbStack, Podman rootless and the userland proxy** hide the real client: every connection reaches
  the container from the Docker bridge gateway (a private `172.x`/`192.168.x` address), so the client filter cannot tell
  your phone from a stranger. Checked on OrbStack: a request from the LAN was seen as the bridge gateway. On such
  hosts the filter does not protect you; **your router must not forward the port**, and `ports:` should name your
  LAN address (`"192.168.1.20:8765:8765"`) rather than all interfaces. On a native Linux Docker engine the source address is
  preserved, so a port forward from the internet is seen as an outside address and refused.
* **Behind a reverse proxy** every request comes from the proxy, a private address, so the filter allows everything the
  proxy lets through. Do not combine the proxy with no password unless the proxy itself requires a login.
* The connection is plain HTTP, as with the password.

## NVIDIA GPU (untested)

Linux: install the NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host, then use `docker-compose.nvidia.yml` (it reserves one GPU). Windows (untested): Docker Desktop with the WSL 2
backend and a current NVIDIA driver on Windows should work with the same file. Check with `docker exec folivox folivox doctor`.

Folivox probes CUDA in a child process before using it and falls back to the CPU if it does not work or the card has
too little memory (4 GB for the voice, 8 GB for the speech check as well), so a misconfigured GPU slows the render
down but does not break it. Use `cuda128` for RTX 50 cards: the older build has no kernels for them.

AMD and Intel GPUs: the optional native runtime has a Vulkan build ([native-runtime.md](native-runtime.md), experimental).
Passing `/dev/dri` into the container might make it usable; this is **untested**, treat it as a guess.

The optional native runtime (CosyVoice, Fish S2 Pro and Drama 3 as GGUF) uses the CUDA build of audio.cpp on an NVIDIA
card, also in a container: NVIDIA's Vulkan driver often fails inside containers (`ERROR_INCOMPATIBLE_DRIVER`), CUDA
does not. The CUDA build needs NCCL, which the `cuda` images already carry with PyTorch. Drama 3 is larger than most
cards (the Q2 file is 17 GB), so part of its expert layers run on the CPU; on a 12 GB card that is about half of them,
and the system needs enough RAM for that part (`folivox engines` shows it). Set `drama3_cpu_expert_layers` under
`[runtime.native]` in `config.toml`, or `FOLIVOX_DRAMA3_CPU_EXPERT_LAYERS`, to override the estimate.

## Update

Pull the new image and recreate the container (Portainer: *Recreate → Re-pull image*; compose:
`docker compose pull && docker compose up -d`). Volumes are kept. A render that is running is cancelled by the stop
(its progress is saved in `/work`) and continues when you start it again.

## Command line in the container

```bash
docker exec -it folivox folivox doctor
# -u: your PUID, so files written are not owned by root; the book must be visible inside the container
docker exec -it -u 1000 folivox folivox render /audiobooks/some.epub
```

## Limits

* No Apple GPU in containers (see the note at the top).
* Books enter through the studio's upload; there is no watched input folder yet.
* The image does not contain the optional Fish Audio engines and native runtime; install them from the studio
  (they download into `/data`).
