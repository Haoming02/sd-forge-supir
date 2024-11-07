import launch

for package in ("openai-clip", "einops-exts", "rich"):
    if not launch.is_installed(package):
        launch.run_pip(f"install {package}", f"{package} for SUPIR")
