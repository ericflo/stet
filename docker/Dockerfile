# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
#
# Build context is the repository ROOT:  docker build -f docker/Dockerfile .
#
# One image: the server, the handbook it renders, and the CLI wheel it hands out. The
# wheel is built from the same commit as the server, so the harness a verifier installs
# from the site is always the one the server expects.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca AS wheel
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv build --wheel -o /out

FROM golang:1.26.11-bookworm@sha256:dc40a13b61304647ed7e5d581d246b86413dfc29d63c6d018310feb93cd98fc0 AS build
WORKDIR /src
ENV CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOFLAGS=-mod=readonly
COPY server/go.mod server/go.sum ./
RUN go mod download
COPY server/*.go server/season.json ./
COPY server/web ./web
ARG STET_REVISION=dev
RUN go build -trimpath -ldflags "-s -w -X main.revision=${STET_REVISION}" -o /out/stet-server .

# Static binary, no shell, no package manager: it runs under restricted pod security.
FROM gcr.io/distroless/static-debian12:nonroot@sha256:afa5c872c891853ca7fcf1f12c3edb23f7eeef36189728842dd51042ff57f7ab
COPY --from=build /out/stet-server /stet-server
COPY --from=wheel /out/ /app/files/
COPY docs/*.md /app/docs/
COPY src/stet/templates/AGENTS.md /app/AGENTS.md
ENV TZ=UTC STET_ADDR=:8080 STET_FILES_DIR=/app/files STET_DOCS_DIR=/app/docs STET_AGENTS_FILE=/app/AGENTS.md STET_DATA_DIR=/tmp/stet
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/stet-server"]
