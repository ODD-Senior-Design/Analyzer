#!/bin/bash

# ===================== config file defaults =====================
CONFIG_FILE=".builddeploy.conf"

# ===================== early arg parse (config file) =====================
for arg in "$@"; do
  case "$arg" in
    --config-file=*) CONFIG_FILE="${arg#*=}" ;;
    -c)
      shift
      CONFIG_FILE="$1" ;;
  esac
done

# ===================== default variable declarations =====================
IMAGE_NAME=""
BUILDER_NAME="multiarch-builder"
PLATFORMS=""
REGISTRY=""
DOCKERFILE_PATH="./Dockerfile"
DOCKER_COMPOSE_COMMAND="sudo docker compose up -d"
SUDO=""

# ===================== load saved config =====================
if [[ -f "$CONFIG_FILE" ]]; then
  source "$CONFIG_FILE"
fi

SUDO="${SUDO:-sudo}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-./Dockerfile}"

if [[ -z "$SUDO" ]]; then
  read -rp "🔐 Do you want to use sudo for Docker commands? [Y/n]: " SUDO_CONFIRM
  if [[ -z "$SUDO_CONFIRM" || "$SUDO_CONFIRM" =~ ^[Yy]$ ]]; then
    SUDO="sudo"
  else
    SUDO=""
  fi
fi

# ===================== flag defaults =====================
SKIP_DEPLOY=false
BUILD_ONLY=false
ALWAYS_REMOVE_BUILDER=false
ALLOW_PRIVATE_REGISTRY=false
LOAD_INSTEAD_OF_PUSH=false
DRY_RUN=false
AUTO_TAG=""
NO_CONFIRM=false
NO_CACHE=false

# ===================== validate_image_name =====================
validate_image_name() {
  local name="$1"
  local dockerhub_pattern='^[a-z0-9]+([._-]?[a-z0-9]+)*/[a-z0-9._-]+(:[a-zA-Z0-9._-]+)?$'
  local registry_pattern='^([a-z0-9.-]+\.[a-z]{2,}/)?[a-z0-9]+([._-]?[a-z0-9]+)*/[a-z0-9._-]+(:[a-zA-Z0-9._-]+)?$'
  if $ALLOW_PRIVATE_REGISTRY; then
    [[ "$name" =~ $registry_pattern ]]
  else
    [[ "$name" =~ $dockerhub_pattern ]]
  fi
}

# ===================== validate_builder_name =====================
validate_builder_name() {
  local name="$1"
  local pattern='^[a-zA-Z0-9._-]+$'
  [[ "$name" =~ $pattern ]]
}

# ===================== dependency checks =====================
for tool in docker; do
  if ! command -v $tool &>/dev/null; then
    echo "❌ Missing required tool: $tool"
    exit 1
  fi
done

if ! $SUDO docker buildx version &>/dev/null; then
  echo "❌ Docker buildx is not installed or not enabled."
  exit 1
fi

# ===================== flag parsing =====================
for arg in "$@"; do
  case "$arg" in
    --no-deploy|-n) SKIP_DEPLOY=true ;;
    --build-only|-b) BUILD_ONLY=true ;;
    --always-rm-builder|-r) ALWAYS_REMOVE_BUILDER=true ;;
    --allow-private-registry|-x) ALLOW_PRIVATE_REGISTRY=true ;;
    --load|-l) LOAD_INSTEAD_OF_PUSH=true ;;
    --dry-run|-d) DRY_RUN=true ;;
    --no-confirm|-y) NO_CONFIRM=true ;;
    --no-cache|-C) NO_CACHE=true ;;
    --no-sudo|-S) SUDO="" ;;
    --dockerfile=*) DOCKERFILE_PATH="${arg#*=}" ;;
    -i)
      shift
      DOCKERFILE_PATH="$1" ;;
    --tag=auto|-t)
      if git rev-parse --short HEAD &>/dev/null; then
        AUTO_TAG="$(git rev-parse --short HEAD)"
      else
        AUTO_TAG="$(date +%Y%m%d%H%M)"
      fi ;;
    --image-name=*) IMAGE_NAME="${arg#*=}" ;;
    --builder-name=*) BUILDER_NAME="${arg#*=}" ;;
    --help|-h)
      echo "📘 Usage: ./buildanddeploy.sh [OPTIONS]"
      echo ""
      echo "Flags:"
      echo "  -n, --no-deploy              Skip docker compose up"
      echo "  -b, --build-only             Only build, don't deploy or remove builder"
      echo "  -r, --always-rm-builder      Always remove builder after build"
      echo "  -x, --allow-private-registry Allow registry.domain.com/image format"
      echo "  -l, --load                   Load image locally instead of pushing"
      echo "  -d, --dry-run                Print all commands but don't execute them"
      echo "  -y, --no-confirm             Skip interactive prompts"
      echo "  -t, --tag=auto               Auto-tag using git SHA or timestamp"
      echo "  -C, --no-cache               Disable Docker build cache"
      echo "  -S, --no-sudo                Run Docker commands without sudo"
      echo "  -i, --dockerfile PATH        Path to Dockerfile (default: ./Dockerfile)"
      echo "  -c, --config-file FILE       Use or create a custom config file"
      echo "      --image-name=NAME        Manually set image name"
      echo "      --builder-name=NAME      Manually set buildx builder name"
      echo "  -h, --help                   Show this help message and exit"
      exit 0 ;;
  esac
done

# ===================== image name prompt =====================
while true; do
  if [[ -z "$IMAGE_NAME" ]]; then
    read -rp "❓ Enter Docker image name (e.g., username/repo:tag): " IMAGE_NAME
  fi

  if [[ "$IMAGE_NAME" != */* ]]; then
    read -rp "🌐 Enter registry (leave blank for Docker Hub): " REGISTRY
    [[ -n "$REGISTRY" ]] && IMAGE_NAME="$REGISTRY/$IMAGE_NAME"
  fi

  [[ -n "$AUTO_TAG" && "$IMAGE_NAME" != *:* ]] && IMAGE_NAME+=":$AUTO_TAG"

  if ! validate_image_name "$IMAGE_NAME"; then
    echo "❌ Invalid image name."
    [[ $ALLOW_PRIVATE_REGISTRY == true ]] && echo "   Format: [registry.domain.com/]username/repo[:tag]" || echo "   Format: username/repo[:tag]"
    IMAGE_NAME=""
    continue
  fi

  PLATFORM_REGEX='^linux/(amd64|arm64|386|arm/v[5-8]|ppc64le|s390x|riscv64)(,linux/(amd64|arm64|386|arm/v[5-8]|ppc64le|s390x|riscv64))*$'
  if [[ -z "$PLATFORMS" ]]; then
    read -rp "🖥️ Enter target platforms (comma-separated). Leave blank for current arch: " PLATFORMS
    if [[ -z "$PLATFORMS" ]]; then
      CURRENT_PLATFORM="$($SUDO docker info --format '{{.OSType}}/{{.Architecture}}' 2>/dev/null || echo "linux/amd64")"
      PLATFORMS="$CURRENT_PLATFORM"
    fi
  fi

  if ! [[ "$PLATFORMS" =~ $PLATFORM_REGEX ]]; then
    echo "❌ Invalid platform(s): '$PLATFORMS'"
    echo "   Supported: linux/amd64, linux/arm64, linux/arm/v7, linux/386, etc."
    IMAGE_NAME=""
    PLATFORMS=""
    continue
  fi

  if [[ -z "$DOCKERFILE_PATH" ]]; then
    read -rp "📦 Enter Dockerfile path [default: ./Dockerfile]: " DOCKERFILE_INPUT
    DOCKERFILE_PATH="${DOCKERFILE_INPUT:-./Dockerfile}"
  fi

  if $NO_CONFIRM; then
    break
  fi

  echo ""
  echo "📝 Final values:"
  echo "   Image:       $IMAGE_NAME"
  echo "   Platforms:   $PLATFORMS"
  echo "   Dockerfile:  $DOCKERFILE_PATH"
  read -rp "🔁 Are these correct? [Y/n]: " CONFIRM
  [[ -z "$CONFIRM" || "$CONFIRM" =~ ^[Yy]$ ]] && break || IMAGE_NAME="" PLATFORMS="" DOCKERFILE_PATH=""
done

# ===================== validate builder name =====================
validate_builder_name "$BUILDER_NAME" || { echo "❌ Invalid builder name: '$BUILDER_NAME'"; exit 1; }

# ===================== save config =====================
cat > "$CONFIG_FILE" <<EOF
IMAGE_NAME="$IMAGE_NAME"
BUILDER_NAME="$BUILDER_NAME"
PLATFORMS="$PLATFORMS"
DOCKERFILE_PATH="$DOCKERFILE_PATH"
DOCKER_COMPOSE_COMMAND="$DOCKER_COMPOSE_COMMAND"
SUDO="$SUDO"
EOF

echo "📏 Configuration saved to '$CONFIG_FILE'"

# ===================== builder setup =====================
CREATED_BUILDER=false

if ! $SUDO docker buildx inspect "$BUILDER_NAME" &>/dev/null; then
  echo "🔧 Setting up buildx for multi-arch support..."
  $SUDO docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
  $SUDO docker buildx create --use --name "$BUILDER_NAME"
  CREATED_BUILDER=true
else
  echo "✅ buildx builder already exists. Using '$BUILDER_NAME'."
  $SUDO docker buildx use "$BUILDER_NAME"
fi

# ===================== bootstrap builder =====================
$SUDO docker buildx inspect --bootstrap

# ===================== build and push =====================
echo ""
echo "🚀 Building image '$IMAGE_NAME'..."

BUILD_CMD="$SUDO docker buildx build --platform \"$PLATFORMS\" -t \"$IMAGE_NAME\" -f \"$DOCKERFILE_PATH\" ."
$LOAD_INSTEAD_OF_PUSH && BUILD_CMD+=" --load" || BUILD_CMD+=" --push"
$NO_CACHE && BUILD_CMD+=" --no-cache"

$DRY_RUN && echo "🔎 Dry-run: $BUILD_CMD" || eval "$BUILD_CMD"

# ===================== remove builder =====================
echo ""
if $ALWAYS_REMOVE_BUILDER; then
  echo "🪝 Removing builder '$BUILDER_NAME'..."
  $SUDO docker buildx rm "$BUILDER_NAME"
else
  echo "ℹ️ Builder '$BUILDER_NAME' was kept."
fi

# ===================== deploy =====================
echo ""
if $BUILD_ONLY || $SKIP_DEPLOY; then
  echo "⏭️  Skipping deployment."
else
  echo "🟢 Starting services with docker compose..."
  $DRY_RUN && echo "🔎 Dry-run: $DOCKER_COMPOSE_COMMAND" || eval "$DOCKER_COMPOSE_COMMAND"
fi
