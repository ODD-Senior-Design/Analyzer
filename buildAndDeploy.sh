#!/bin/bash

# ===================== config file defaults =====================
CONFIG_FILE=".builddeploy.conf"

# ===================== early arg parse (config file) =====================
for arg in "$@"; do
  case "$arg" in
    --config-file=*)
      CONFIG_FILE="${arg#*=}"
      ;;
    -c)
      shift
      CONFIG_FILE="$1"
      ;;
  esac
done

# ===================== default variable declarations =====================
IMAGE_NAME=""
BUILDER_NAME="multiarch-builder"
PLATFORMS=""
REGISTRY=""
DOCKER_COMPOSE_COMMAND="sudo docker compose up -d"
SUDO=""

# ===================== load saved config =====================
if [[ -f "$CONFIG_FILE" ]]; then
  source "$CONFIG_FILE"
fi

# Prompt to choose sudo if not set by config or CLI
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
      echo "  -c, --config-file FILE       Use custom config file"
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

  if [[ "$IMAGE_NAME" != *"/"* ]]; then
    read -rp "🌐 Enter registry (leave blank for Docker Hub): " REGISTRY
    if [[ -n "$REGISTRY" ]]; then
      IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}"
    fi
  fi

  if [[ -n "$AUTO_TAG" && "$IMAGE_NAME" != *:* ]]; then
    IMAGE_NAME="${IMAGE_NAME}:${AUTO_TAG}"
  fi

  if ! validate_image_name "$IMAGE_NAME"; then
    echo "❌ Invalid image name."
    if $ALLOW_PRIVATE_REGISTRY; then
      echo "   Format: [registry.domain.com/]username/repo[:tag]"
    else
      echo "   Format: username/repo[:tag]"
    fi
    IMAGE_NAME=""
    continue
  fi

  PLATFORM_REGEX='^linux/(amd64|arm64|386|arm/v[5-8]|ppc64le|s390x|riscv64)(,linux/(amd64|arm64|386|arm/v[5-8]|ppc64le|s390x|riscv64))*$'
  if [[ -z "$PLATFORMS" ]]; then
    read -rp "🖥️ Enter target platforms (comma-separated, e.g., linux/amd64,linux/arm64). Leave blank for current arch: " PLATFORMS
    if [[ -z "$PLATFORMS" ]]; then
      if CURRENT_PLATFORM=$($SUDO docker info --format '{{.OSType}}/{{.Architecture}}' 2>/dev/null); then
        PLATFORMS="$CURRENT_PLATFORM"
      else
        PLATFORMS="linux/amd64"
      fi
    fi
  fi

  if ! [[ "$PLATFORMS" =~ $PLATFORM_REGEX ]]; then
    echo "❌ Invalid platform(s): '$PLATFORMS'"
    echo "   Supported: linux/amd64, linux/arm64, linux/arm/v7, linux/386, etc."
    IMAGE_NAME=""
    PLATFORMS=""
    continue
  fi

  if $NO_CONFIRM; then
    break
  fi

  echo ""
  echo "📝 Final values:"
  echo "   Image:     $IMAGE_NAME"
  echo "   Platforms: $PLATFORMS"
  read -rp "🔁 Are these correct? [Y/n]: " CONFIRM
  if [[ -z "$CONFIRM" || "$CONFIRM" =~ ^[Yy]$ ]]; then
    break
  else
    IMAGE_NAME=""
    PLATFORMS=""
  fi
done

# ===================== validate builder name =====================
if ! validate_builder_name "$BUILDER_NAME"; then
  echo "❌ Invalid builder name: '$BUILDER_NAME'. Only letters, numbers, '.', '_', and '-' are allowed."
  exit 1
fi

# ===================== save config =====================
cat > "$CONFIG_FILE" <<EOF
IMAGE_NAME="$IMAGE_NAME"
BUILDER_NAME="$BUILDER_NAME"
PLATFORMS="$PLATFORMS"
DOCKER_COMPOSE_COMMAND="$DOCKER_COMPOSE_COMMAND"
SUDO="$SUDO"
EOF
echo "💾 Configuration saved to '$CONFIG_FILE'"

# ===================== builder setup =====================
CREATED_BUILDER=false
echo ""
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

BUILD_CMD="$SUDO docker buildx build --platform \"$PLATFORMS\" -t \"$IMAGE_NAME\" ."
if $LOAD_INSTEAD_OF_PUSH; then
  BUILD_CMD+=" --load"
else
  BUILD_CMD+=" --push"
fi
if $NO_CACHE; then
  BUILD_CMD+=" --no-cache"
fi

if $DRY_RUN; then
  echo "🔎 Dry-run: $BUILD_CMD"
else
  eval "$BUILD_CMD"
fi

# ===================== remove builder =====================
echo ""
if $ALWAYS_REMOVE_BUILDER; then
  echo "🧹 Removing builder '$BUILDER_NAME'..."
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
  if $DRY_RUN; then
    echo "🔎 Dry-run: $DOCKER_COMPOSE_COMMAND"
  else
    eval "$DOCKER_COMPOSE_COMMAND"
  fi
fi
