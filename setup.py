"""Setup file for PIMO."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

from setuptools import find_packages, setup


def load_module(name: str) -> ModuleType:
    """Load Python Module."""
    location = str(Path(__file__).parent / name)
    spec = spec_from_file_location(name=name, location=location)
    module = module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def get_version() -> str:
    """Get version from `aupimo.__version__.py`."""
    version = load_module(name="src/aupimo/__version__.py")
    return version.__version__


def get_required_packages(requirement_files: list[str]) -> list[str]:
    """Get packages from requirements.txt file."""
    required_packages: list[str] = []
    for requirement_file in requirement_files:
        with Path(f"requirements/{requirement_file}.txt").open(encoding="utf8") as file:
            for line in file:
                package = line.strip()
                if package and not package.startswith(("#", "-f")):
                    required_packages.append(package)
    return required_packages


LONG_DESCRIPTION = (Path(__file__).parent / "README.md").read_text(encoding="utf8")

setup(
    name="aupimo",
    version=get_version(),
    author="jpcbertoldo",
    description="AUPIMO: Area Under the Per-IMage Overlap curve.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="",
    license_files=('LICENSE',),
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["aupimo"]),
    install_requires=get_required_packages(requirement_files=["base"]),
    include_package_data=True,
    package_data={"": ["config.yaml"]},
)
