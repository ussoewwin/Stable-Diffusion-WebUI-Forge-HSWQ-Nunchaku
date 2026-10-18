"""
This script installs necessary requirements and launches main program in webui.py
"""

import importlib.metadata
import importlib.util
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

from modules import cmd_args, errors, logging_config
from modules.paths_internal import extensions_builtin_dir, extensions_dir, script_path
from modules.timer import startup_timer
from modules_forge import forge_version
from modules_forge.config import always_disabled_extensions

args, _ = cmd_args.parser.parse_known_args()
logging_config.setup_logging(args.loglevel)

python = sys.executable
git = os.environ.get("GIT", "git")
index_url = os.environ.get("INDEX_URL", "")
dir_repos = "repositories"

default_command_live = os.environ.get("WEBUI_LAUNCH_LIVE_OUTPUT") == "1"

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("NUMPY_WARN_IF_NO_MEM_POLICY", "0")


def check_python_version():
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if not (major == 3 and minor == 11):
        import modules.errors

        modules.errors.print_error_explanation(
            f"""
            This program is tested with 3.11.9 Python, but you have {major}.{minor}.{micro}.
            If you encounter any error regarding unsuccessful package/library installation,
            please downgrade (or upgrade) to the latest version of 3.11 Python,
            and delete the current Python "venv" folder in WebUI's directory.

            Use --skip-python-version-check to suppress this warning
            """
        )


def git_tag():
    return forge_version.version


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": "utf8",
        "errors": "ignore",
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return result.stdout or ""


def _torch_version() -> tuple[str, str]:
    """Given `2.10.0.dev20251111+cu130` ; Return `("2.10.0", "cu130")`"""
    import importlib.metadata

    ver = importlib.metadata.version("torch")
    m = re.search(r"(\d+\.\d+\.\d+)(?:[^+]+)?\+(.+)", ver)

    if m is None:
        print("\n\nFailed to parse PyTorch version...")
        ver = os.environ.get("PYTORCH_VERSION", "2.13.0+cu132")
        print("Assuming: ", ver)
        print('(you can change this with `export PYTORCH_VERSION="..."`)\n\n')
        m = re.search(r"(\d+\.\d+\.\d+)(?:[^+]+)?\+(.+)", ver)

    return m.group(1), m.group(2)


def is_installed(package):
    try:
        dist = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            return False

        return spec is not None

    return dist is not None


def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


def run_pip(command, desc=None, live=default_command_live):
    if args.skip_install:
        return

    index_url_line = f" --index-url {index_url}" if index_url != "" else ""
    # NumPy 2.2.6 compatibility: Remove --prefer-binary for Pillow to avoid metadata issues
    prefer_binary_flag = "" if desc and "pillow" in desc.lower() else " --prefer-binary"
    return run(f'"{python}" -m pip {command}{prefer_binary_flag}{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def _torch_stack_constraint_file() -> str | None:
    """
    Pin only installed torch (exact local version, e.g. 2.13.0+cu132).
    Do not pin torchvision/torchaudio here: their metadata often requires a different torch
    (e.g. torchvision 0.27.1+cu132 -> torch==2.12.1), and pinning all three makes
    `pip install -r requirements.txt` fail with ResolutionImpossible.
    """
    import importlib.metadata

    try:
        ver = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None

    path = os.path.join(script_path, "tmp", "torch_stack_constraints.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"torch=={ver}\n")
    return path


def _apply_torch_stack_pip_constraint() -> None:
    """
    Pin installed torch for subsequent pip installs (including extension install.py).
    Prevents PyPI CPU torch from replacing a CUDA build during `pip install -U` / deps.
    """
    path = _torch_stack_constraint_file()
    if path is None:
        return
    os.environ["PIP_CONSTRAINT"] = path
    print(f"Torch pinned ({path}); pip will not replace torch")


def check_run_python(code: str, *, return_error: bool = False) -> bool | tuple[bool, str]:
    result = subprocess.run([python, "-c", code], capture_output=True, shell=False)
    if return_error:
        return result.returncode == 0, result.stderr
    else:
        return result.returncode == 0


def git_fix_workspace(*args, **kwargs):
    raise NotImplementedError()


def run_git(*args, **kwargs):
    raise NotImplementedError()


def git_clone(*args, **kwargs):
    raise NotImplementedError()


def git_pull_recursive(dir):
    for subdir, _, _ in os.walk(dir):
        if os.path.exists(os.path.join(subdir, ".git")):
            try:
                output = subprocess.check_output([git, "-C", subdir, "pull", "--autostash"])
                print(f"Pulled changes for repository in '{subdir}':\n{output.decode('utf-8').strip()}\n")
            except subprocess.CalledProcessError as e:
                print(f"Couldn't perform 'git pull' on repository in '{subdir}':\n{e.output.decode('utf-8').strip()}\n")


def run_extension_installer(extension_dir):
    path_installer = os.path.join(extension_dir, "install.py")
    if not os.path.isfile(path_installer):
        return

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{script_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

        stdout = run(f'"{python}" "{path_installer}"', errdesc=f"Error running install.py for extension {extension_dir}", custom_env=env).strip()
        if stdout:
            print(stdout)
    except Exception as e:
        errors.report(str(e))


def list_extensions(settings_file):
    settings = {}

    try:
        with open(settings_file, "r", encoding="utf8") as file:
            settings = json.load(file)
    except FileNotFoundError:
        pass
    except Exception:
        errors.report(f'\nCould not load settings\nThe config file "{settings_file}" is likely corrupted\nIt has been moved to the "tmp/config.json"\nReverting config to default\n\n' "", exc_info=True)
        os.replace(settings_file, os.path.join(script_path, "tmp", "config.json"))

    disabled_extensions = set(settings.get("disabled_extensions", []) + always_disabled_extensions)
    disable_all_extensions = settings.get("disable_all_extensions", "none")

    if disable_all_extensions != "none" or args.disable_extra_extensions or args.disable_all_extensions or not os.path.isdir(extensions_dir):
        return []

    builtin_names = set()
    if os.path.isdir(extensions_builtin_dir):
        builtin_names = set(os.listdir(extensions_builtin_dir))
    return [x for x in os.listdir(extensions_dir) if x not in disabled_extensions and x not in builtin_names]


def list_extensions_builtin(settings_file):
    settings = {}

    try:
        with open(settings_file, "r", encoding="utf8") as file:
            settings = json.load(file)
    except FileNotFoundError:
        pass
    except Exception:
        errors.report(f'\nCould not load settings\nThe config file "{settings_file}" is likely corrupted\nIt has been moved to the "tmp/config.json"\nReverting config to default\n\n' "", exc_info=True)
        os.replace(settings_file, os.path.join(script_path, "tmp", "config.json"))

    disabled_extensions = set(settings.get("disabled_extensions", []))
    disable_all_extensions = settings.get("disable_all_extensions", "none")

    if disable_all_extensions != "none" or args.disable_extra_extensions or args.disable_all_extensions or not os.path.isdir(extensions_builtin_dir):
        return []

    return [x for x in os.listdir(extensions_builtin_dir) if x not in disabled_extensions]


def run_extensions_installers(settings_file):
    if os.path.isdir(extensions_dir):
        with startup_timer.subcategory("run extensions installers"):
            for dirname_extension in list_extensions(settings_file):
                logging.debug(f"Installing {dirname_extension}")

                path = os.path.join(extensions_dir, dirname_extension)

                if os.path.isdir(path):
                    run_extension_installer(path)
                    startup_timer.record(dirname_extension)

    if os.path.isdir(extensions_builtin_dir):
        with startup_timer.subcategory("run extensions_builtin installers"):
            for dirname_extension in list_extensions_builtin(settings_file):
                logging.debug(f"Installing {dirname_extension}")

                path = os.path.join(extensions_builtin_dir, dirname_extension)

                if os.path.isdir(path):
                    run_extension_installer(path)
                    startup_timer.record(dirname_extension)

    return


re_requirement = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(?:(==|>=|>|<=|<)\s*(\S+))?")


def requirements_met(requirements_file):
    """
    Does a simple parse of a requirements.txt file to determine if all rerqirements in it
    are already installed. Returns True if so, False if not installed or parsing fails.
    Supports ==, >=, >, <=, < version specifiers.
    """

    import importlib.metadata

    import packaging.version

    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if (m := re.match(re_requirement, line)) is None:
                continue

            package = m.group(1)
            op = m.group(2)
            version_required = m.group(3)

            if op is None or version_required is None:
                if not is_installed(package):
                    return False
                continue

            try:
                version_installed = importlib.metadata.version(package)
                installed = packaging.version.parse(version_installed)
                required = packaging.version.parse(version_required)
            except Exception:
                return False

            if op == "==":
                if installed != required:
                    return False
            elif op == ">=":
                if installed < required:
                    return False
            elif op == ">":
                if installed <= required:
                    return False
                elif op == "<=":
                    if installed > required:
                        return False
                elif op == "<":
                    if installed >= required:
                        return False

    return True


def prepare_environment():
    torch_index_url = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu132")
    # Initial-install defaults only. Already-installed packages (including newer builds) are left alone.
    torch_default = os.environ.get("TORCH_PACKAGE", "torch")
    torchvision_default = os.environ.get("TORCHVISION_PACKAGE", "torchvision")
    torch_command = os.environ.get(
        "TORCH_COMMAND",
        f"pip install {torch_default} {torchvision_default} --index-url {torch_index_url}",
    )
    xformers_package = os.environ.get("XFORMERS_PACKAGE", f"xformers==0.0.33.post2 --extra-index-url {torch_index_url}")
    bnb_package = os.environ.get("BNB_PACKAGE", "bitsandbytes==0.48.2")

    clip_package = os.environ.get("CLIP_PACKAGE", "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip")
    packaging_package = os.environ.get("PACKAGING_PACKAGE", "packaging==25.0")
    gradio_package = os.environ.get(
        "GRADIO_PACKAGE",
        "https://huggingface.co/ussoewwin/gradio/resolve/main/gradio-4.40.0-py3-none-any.whl gradio_imageslider==0.0.20 gradio_rangeslider==0.0.8",
    )
    requirements_file = os.environ.get("REQS_FILE", "requirements.txt")

    try:
        # the existence of this file is a signal to webui.sh/bat that webui needs to be restarted when it stops execution
        os.remove(os.path.join(script_path, "tmp", "restart"))
        os.environ.setdefault("SD_WEBUI_RESTARTING", "1")
    except OSError:
        pass

    if not args.skip_python_version_check:
        check_python_version()

    startup_timer.record("checks")

    tag = git_tag()

    print(f"Python {sys.version}")
    print(f"Version: {tag}")

    # Per-package: install default stack only when missing.
    # If torch / torchvision is already installed (including newer), do nothing to that package.
    # --reinstall-torch forces the full default stack.
    if args.reinstall_torch:
        os.environ.pop("PIP_CONSTRAINT", None)
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)
        startup_timer.record("install torch")
    else:
        torch_pkgs = []
        if not is_installed("torch"):
            torch_pkgs.append(torch_default)
        if not is_installed("torchvision"):
            torch_pkgs.append(torchvision_default)
        if torch_pkgs:
            os.environ.pop("PIP_CONSTRAINT", None)
            run(
                f'"{python}" -m pip install {" ".join(torch_pkgs)} --index-url {torch_index_url}',
                "Installing missing torch stack packages",
                "Couldn't install torch stack",
                live=True,
            )
            startup_timer.record("install torch")

    # After torch stack is settled, pin it so later pip -U / deps cannot swap in PyPI CPU torch.
    _apply_torch_stack_pip_constraint()

    if not args.skip_torch_cuda_test:
        success, err = check_run_python("import torch; assert torch.cuda.is_available()", return_error=True)
        if not success:
            if "older driver" in str(err).lower():
                raise SystemError("Please update your GPU driver to support cu132 ; or manually install older PyTorch")
            raise RuntimeError("PyTorch is not able to access CUDA")
        startup_timer.record("torch GPU test")

    if not is_installed("packaging"):
        run_pip(f"install {packaging_package}", "packaging")

    distutils_package = os.environ.get(
        "DISTUTILS_PACKAGE",
        "https://huggingface.co/ussoewwin/distutils/resolve/main/distutils-3.14.0-py3-none-any.whl",
    )
    if not is_installed("distutils"):
        try:
            run_pip(f"install {distutils_package}", "distutils")
        except RuntimeError:
            print("Failed to install distutils; Please manually install it")
        else:
            startup_timer.record("install distutils")

    pyaudioop_package = os.environ.get("PYAUDIOOP_PACKAGE", "https://huggingface.co/ussoewwin/pyaudioop-1.0.0-py3-none-any/resolve/main/pyaudioop-1.0.0-py3-none-any.whl")
    if not is_installed("pyaudioop"):
        run_pip(f"install {pyaudioop_package}", "pyaudioop")
        startup_timer.record("install pyaudioop")

    if not is_installed("insightface"):
        run_pip("install insightface", "insightface")
        startup_timer.record("install insightface")

    facexlib_package = os.environ.get(
        "FACEXLIB_PACKAGE",
        "https://huggingface.co/ussoewwin/facexlib/resolve/main/facexlib-0.3.0-py3-none-any.whl",
    )
    if not is_installed("facexlib"):
        try:
            run_pip(f"install {facexlib_package} --no-deps", "facexlib")
        except RuntimeError:
            print("Failed to install facexlib; Please manually install it")
        else:
            startup_timer.record("install facexlib")

    ver_PY = f"cp{sys.version_info.major}{sys.version_info.minor}"
    ver_SAGE = "2.2.0"
    ver_FLASH = "2.8.4" if sys.version_info >= (3, 14) else "2.8.3"
    ver_TRITON = "3.5.1"
    ver_NUNCHAKU = "1.1.0"
    ver_TORCH, ver_CUDA = _torch_version()
    v_TORCH = ver_TORCH.rsplit(".", 1)[0]

    if os.name == "nt":
        if sys.version_info >= (3, 14):
            default_sage_package = "https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattention-2.2.0.post6%2Bcu132torch2.13.0-cp314-cp314-win_amd64.whl"
            default_sageattn3_package = "https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattn3-1.0.0%2Bcu132torch2.13.0-cp314-cp314-win_amd64.whl"
        else:
            default_sage_package = "https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattention-2.2.0%2Bcu130torch2.11.0-cp313-cp313-win_amd64.whl"
            default_sageattn3_package = "https://huggingface.co/ussoewwin/Sage-Attention-for-Windows/resolve/main/sageattn3-1.0.0%2Bcu130torch2.11.0-cp313-cp313-win_amd64.whl"

        sage_package = os.environ.get(
            "SAGE_PACKAGE",
            default_sage_package,
        )
        sageattn3_package = os.environ.get(
            "SAGEATTN3_PACKAGE",
            default_sageattn3_package,
        )
        if sys.version_info >= (3, 14):
            default_flash_package = "https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.8.4%2Bcu132torch2.13.0cxx11abiTRUE-cp314-cp314-win_amd64.whl"
        else:
            default_flash_package = "https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.8.3%2Bcu130torch2.11.0cxx11abiTRUE-cp313-cp313-win_amd64.whl"

        flash_package = os.environ.get(
            "FLASH_PACKAGE",
            default_flash_package,
        )
        triton_package = os.environ.get("TRITION_PACKAGE", "triton-windows")
        if sys.version_info >= (3, 14):
            default_nunchaku_package = "https://huggingface.co/ussoewwin/nunchaku-built-on-cu132/resolve/main/nunchaku-1.3.0.dev20260619%2Bcu13.2torch2.12-cp314-cp314-win_amd64.whl"
        else:
            default_nunchaku_package = f"https://github.com/nunchaku-tech/nunchaku/releases/download/v{ver_NUNCHAKU}/nunchaku-{ver_NUNCHAKU}+torch{v_TORCH}-{ver_PY}-{ver_PY}-win_amd64.whl"

        nunchaku_package = os.environ.get(
            "NUNCHAKU_PACKAGE",
            default_nunchaku_package,
        )

    else:
        sage_package = os.environ.get("SAGE_PACKAGE", f"sageattention=={ver_SAGE}")
        sageattn3_package = os.environ.get("SAGEATTN3_PACKAGE", None)
        flash_package = os.environ.get("FLASH_PACKAGE", f"https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-{ver_FLASH}+{ver_CUDA}torch{v_TORCH}-{ver_PY}-{ver_PY}-linux_x86_64.whl")
        triton_package = os.environ.get("TRITION_PACKAGE", f"triton=={ver_TRITON}")
        if sys.version_info >= (3, 14):
            default_nunchaku_linux = "https://huggingface.co/ussoewwin/nunchaku-built-on-cu132/resolve/main/nunchaku-1.3.0.dev20260829%2Bcu13.2torch2.13-cp314-cp314-linux_x86_64.whl"
        else:
            default_nunchaku_linux = f"https://github.com/nunchaku-tech/nunchaku/releases/download/v{ver_NUNCHAKU}/nunchaku-{ver_NUNCHAKU}+torch{v_TORCH}-{ver_PY}-{ver_PY}-linux_x86_64.whl"
        nunchaku_package = os.environ.get("NUNCHAKU_PACKAGE", default_nunchaku_linux)

    def _verify_native_import(package: str) -> bool:
        """Verify a package can actually be imported (catches DLL / native extension failures)."""
        if not is_installed(package):
            return False
        return check_run_python(f"import {package}")

    def _force_uninstall(package: str) -> None:
        """Uninstall a package, force-removing residual files if pip fails (e.g. locked .pyd on Windows)."""
        import shutil
        import sysconfig
        site_packages = sysconfig.get_paths()["purelib"]
        try:
            run(f'"{python}" -m pip uninstall -y {package}', f"Uninstalling broken {package}")
        except RuntimeError:
            print(f"pip uninstall failed for {package}; attempting manual removal")
        for entry in os.listdir(site_packages):
            if entry.startswith(package) or entry.startswith(package.replace("-", "_")):
                path = os.path.join(site_packages, entry)
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except OSError as e:
                    print(f"Cannot remove {path}: {e} — close other Python processes and retry")

    def _verify_nunchaku() -> bool:
        if not is_installed("nunchaku"):
            return False

        import importlib.metadata

        import packaging.version

        ver_installed: str = importlib.metadata.version("nunchaku")
        current: tuple[int] = packaging.version.parse(ver_installed)
        target: tuple[int] = packaging.version.parse(ver_NUNCHAKU)

        return current >= target

    if not is_installed("clip"):
        # Ensure setuptools (pkg_resources) is available for CLIP's setup.py on Python 3.13+
        if not is_installed("setuptools"):
            run_pip("install setuptools<82", "setuptools (for clip)")
        run_pip(f"install --no-build-isolation {clip_package}", "clip")
        startup_timer.record("install clip")

    if args.xformers and (not is_installed("xformers") or args.reinstall_xformers):
        run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")
        startup_timer.record("install xformers")

    if args.sage:
        if not is_installed("triton"):
            try:
                run_pip(f"install -U -I --no-deps {triton_package}", "triton")
            except RuntimeError:
                print("Failed to install triton; Please manually install it")
            else:
                startup_timer.record("install triton")
        if not _verify_native_import("sageattention"):
            _force_uninstall("sageattention")
            try:
                run_pip(f"install --no-deps {sage_package}", "sageattention")
            except RuntimeError:
                print("Failed to install sageattention; Please manually install it")
            else:
                startup_timer.record("install sageattention")
        if sageattn3_package and not _verify_native_import("sageattn3"):
            _force_uninstall("sageattn3")
            try:
                run_pip(f"install --no-deps {sageattn3_package}", "sageattn3")
            except RuntimeError:
                print("Failed to install sageattn3; Please manually install it")
            else:
                startup_timer.record("install sageattn3")

    if args.flash and not _verify_native_import("flash_attn"):
        _force_uninstall("flash_attn")
        try:
            run_pip(f"install --no-deps {flash_package}", "flash_attn")
        except RuntimeError:
            print("Failed to install flash_attn; Please manually install it")
        else:
            startup_timer.record("install flash_attn")

    if args.nunchaku and not _verify_nunchaku():
        try:
            run_pip(f"install --no-deps {nunchaku_package}", "nunchaku")
        except RuntimeError:
            print("Failed to install nunchaku; Please manually install it")
        else:
            startup_timer.record("install nunchaku")

    if not is_installed("iopaint"):
        try:
            run_pip("install iopaint --no-deps", "iopaint")
        except RuntimeError:
            print("Failed to install iopaint; Please manually install it")
        else:
            startup_timer.record("install iopaint")

    if sys.version_info >= (3, 13) and not is_installed("imghdr"):
        try:
            run_pip(
                "install https://huggingface.co/ussoewwin/imghdr-1.0.0-py3-none-any/resolve/main/imghdr-1.0.0-py3-none-any.whl",
                "imghdr",
            )
        except RuntimeError:
            print("Failed to install imghdr; Please manually install it")
        else:
            startup_timer.record("install imghdr")

    if args.bnb and not is_installed("bitsandbytes"):
        try:
            run_pip(f"install {bnb_package}", "bitsandbytes")
        except RuntimeError:
            print("Failed to install bitsandbytes; Please manually install it")
        else:
            startup_timer.record("install bitsandbytes")

    if not is_installed("ngrok") and args.ngrok:
        run_pip("install ngrok", "ngrok")
        startup_timer.record("install ngrok")

    if not is_installed("gradio"):
        run_pip(f"install {gradio_package}", "gradio")

    if not os.path.isfile(requirements_file):
        requirements_file = os.path.join(script_path, requirements_file)

    if args.onnxruntime_gpu and not is_installed("onnxruntime-gpu"):
        try:
            run_pip(
                "install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu",
                "onnxruntime-gpu",
            )
        except RuntimeError:
            print("Failed to install onnxruntime-gpu; Please manually install it")
        else:
            startup_timer.record("install onnxruntime-gpu")

    if args.update_all_extensions:
        git_pull_recursive(extensions_dir)
        startup_timer.record("update extensions")

    run_extensions_installers(settings_file=args.ui_settings_file)

    if not requirements_met(requirements_file):
        # only-if-needed: do not upgrade already-satisfied deps (esp. torch pulled by accelerate/etc.)
        run_pip(f'install -U --upgrade-strategy only-if-needed -r "{requirements_file}"', "requirements")
        startup_timer.record("enforce requirements")

    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        exit(0)


class ModelRef(NamedTuple):
    arg_name: str
    relative_path: str


def configure_a1111_reference(a1111_home: Path):
    """Append model paths based on an existing A1111 installation"""

    refs = (
        ModelRef(arg_name="--embeddings-dir", relative_path="embeddings"),
        ModelRef(arg_name="--esrgan-models-path", relative_path="ESRGAN"),
        ModelRef(arg_name="--lora-dirs", relative_path="Lora"),
        ModelRef(arg_name="--ckpt-dirs", relative_path="Stable-diffusion"),
        ModelRef(arg_name="--text-encoder-dirs", relative_path="text_encoder"),
        ModelRef(arg_name="--vae-dirs", relative_path="VAE"),
        ModelRef(arg_name="--controlnet-dir", relative_path="ControlNet"),
        ModelRef(arg_name="--controlnet-preprocessor-models-dir", relative_path="ControlNetPreprocessor"),
    )

    for ref in refs:
        target_path = a1111_home / ref.relative_path
        if not target_path.exists():
            target_path = a1111_home / "models" / ref.relative_path
        if not target_path.exists():
            continue

        sys.argv.extend([ref.arg_name, str(target_path.absolute())])


def configure_comfy_reference(comfy_home: Path):
    """Append model paths based on an existing Comfy installation"""

    refs = (
        ModelRef(arg_name="--ckpt-dirs", relative_path="checkpoints"),
        ModelRef(arg_name="--ckpt-dirs", relative_path="diffusion_models"),
        ModelRef(arg_name="--ckpt-dirs", relative_path="unet"),
        ModelRef(arg_name="--text-encoder-dirs", relative_path="clip"),
        ModelRef(arg_name="--text-encoder-dirs", relative_path="text_encoders"),
        ModelRef(arg_name="--lora-dirs", relative_path="loras"),
        ModelRef(arg_name="--vae-dirs", relative_path="vae"),
    )

    for ref in refs:
        target_path = comfy_home / ref.relative_path
        if not target_path.exists():
            target_path = comfy_home / "models" / ref.relative_path
        if not target_path.exists():
            continue

        sys.argv.extend([ref.arg_name, str(target_path.absolute())])


def start():
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {shlex.join(sys.argv[1:])}")
    # Install import hook so first "import transformers" gets HybridCache patched (no early import)
    from modules_forge import transformers_cache_compat
    transformers_cache_compat.apply()
    import webui

    if "--nowebui" in sys.argv:
        webui.api_only()
    else:
        webui.webui()

    from modules_forge import main_thread

    main_thread.loop()
    return


def dump_sysinfo():
    import datetime

    from modules import sysinfo

    text = sysinfo.get()
    filename = f"sysinfo-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d-%H-%M')}.json"

    with open(filename, "w", encoding="utf8") as file:
        file.write(text)

    return filename
