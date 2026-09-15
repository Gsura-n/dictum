#!/usr/bin/env bash
# Deterministic fixture: a home folder and a project folder with the kinds of
# files spoken commands tend to be about. Same bytes and mtimes every build.
set -euo pipefail
F=${1:-/fixture}
rm -rf "$F"; mkdir -p "$F"
H=$F/home; W=$F/work
mkdir -p $H/Downloads $H/Documents $H/Desktop
mkdir -p $W/src/utils $W/src/__pycache__ $W/logs $W/data $W/images $W/nested/a/b/c/d/e/f $W/build $W/empty_dir $W/tmp

truncate -s 120M $H/Downloads/ubuntu.iso
truncate -s 15M  $H/Downloads/video.mp4
printf 'invoice\n' > $H/Downloads/invoice.pdf
printf 'meeting notes\nTODO call vendor\n' > $H/Documents/notes.txt
printf 'draft\n' > $H/Desktop/draft.txt

printf 'import os\n\ndef main():\n    print("hello")  # TODO\n\nif __name__ == "__main__":\n    main()\n' > $W/src/main.py
printf 'def add(a, b):\n    return a + b\n' > $W/src/utils/helpers.py
printf '' > $W/src/utils/__init__.py
printf 'x' > $W/src/__pycache__/main.cpython-312.pyc
printf 'x' > $W/src/utils/helpers.pyc
printf 'console.log("todo");\n' > $W/src/app.js
for i in $(seq 1 200); do printf '2026-09-01 10:%02d:00 INFO request %d ok\n' $((i%60)) $i; done > $W/logs/app.log
printf '2026-09-01 ERROR disk full\n2026-09-01 ERROR timeout\n' > $W/logs/error.log
printf 'old\n' > $W/logs/old.log
printf 'id,region,amount\n1,west,100\n2,east,250\n3,west,75\n' > $W/data/sales.csv
printf 'id,value\n1,a\n' > $W/data/2024.csv
printf '# Data\n' > $W/data/README.md
printf 'raw' > $W/images/photo.NEF
printf 'raw' > $W/images/other.NEF
seq 1 20000 > $W/dump.txt
printf 'col1\tcol2\tcol3\nalpha\tbeta\tgamma\n' > $W/file.txt
printf 'junk\n' > $W/junk; printf 'dummy\n' > $W/dummy
printf 'deep\n' > $W/nested/a/b/c/d/e/f/deep.txt
printf '#!/bin/sh\necho run\n' > $W/script.sh; chmod +x $W/script.sh
printf 'secret\n' > $W/.env
printf 'c45\n' > $W/c45_data.txt; printf 'c80\n' > $W/c80_data.txt
printf 'a\nb\na\nc\nb\na\n' > $W/letters.txt
ln -s src/main.py $W/main_link.py
ln -s does_not_exist.txt $W/broken_link
chmod 600 $W/data/sales.csv

# Fixed mtimes: most files old, a few recent, so "modified in the last N days" has an answer.
find $F -exec touch -h -d '2026-01-15 12:00:00' {} +
touch -h -d '2026-09-10 09:00:00' $W/logs/app.log $W/src/main.py $H/Documents/notes.txt
