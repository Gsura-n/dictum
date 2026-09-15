# Throwaway environment for execution-based command scoring.
# Runs with --network none; every command gets a fresh copy of /fixture.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 bzip2 zip unzip xz-utils file bc gawk tree rsync less procps psmisc lsof \
      coreutils findutils grep sed tar gzip diffutils util-linux ncurses-bin \
    && rm -rf /var/lib/apt/lists/*
# Commands that reach for sudo should not fail just because it is missing.
RUN printf '#!/bin/sh\nexec "$@"\n' > /usr/local/bin/sudo && chmod +x /usr/local/bin/sudo
COPY build_fixture.sh runner.py /opt/sandbox/
RUN bash /opt/sandbox/build_fixture.sh /fixture && mkdir -p /runs
CMD ["sleep", "infinity"]
