#!/bin/sh
# Build the Swift version without Xcode (Command Line Tools only).
#
# MLX Swift compiles its GPU kernels with Apple's Metal compiler, which comes with Xcode. Without it, this script
# builds everything else and puts the same kernels, prebuilt, next to the programs: the shader library
# (mlx.metallib) from the official MLX Python wheel of the same MLX version. MLX looks for it there first.
# With Xcode installed, use `swift build -c release` or Xcode instead.
set -eu
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# stand-ins for the Metal compiler and linker: empty outputs, so the build can finish
cat > "$tmp/metal" <<'EOF'
#!/bin/sh
out=""; dep=""
while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift;; -MF) dep="$2"; shift;; esac; shift; done
[ -n "$out" ] && : > "$out"
[ -n "$dep" ] && echo "dependencies:" > "$dep"
exit 0
EOF
cp "$tmp/metal" "$tmp/metallib"
chmod +x "$tmp/metal" "$tmp/metallib"
PATH="$tmp:$PATH" swift build -c release

h=.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/version.h
v=$(awk '/#define MLX_VERSION_(MAJOR|MINOR|PATCH) / { printf "%s%s", sep, $3; sep = "." }' "$h")
echo "MLX $v: fetching its prebuilt shader library (the mlx-metal $v wheel)"
# --python-version: the wheel needs Python 3.10+ to install, but here it is only unzipped, so macOS's own
# python3 (3.9) can fetch it
python3 -m pip download --quiet --disable-pip-version-check --no-deps --only-binary=:all: --python-version 3.12 \
    "mlx-metal==$v" -d "$tmp/wheel"
unzip -q -o -j "$tmp"/wheel/mlx_metal-*.whl 'mlx/lib/mlx.metallib' -d .build/release/
echo "built .build/release/fdmlx-swift (needs .build/release/mlx.metallib next to it)"
