```markdown
# SpikeInterface Demo Setup

This repository contains Jupyter Notebook demos for spike sorting using the [SpikeInterface](https://spikeinterface.readthedocs.io/en/latest/) toolkit. Follow the instructions below to set up your environment and run the demos on macOS, Linux, or Windows.

---

## 📦 Prerequisites (All Platforms)

### 1. Install Miniconda (Recommended)
Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your OS.

- During installation, check "Add Miniconda to your PATH".
- Verify installation:
  ```bash
  conda --version
  ```

### 2. Verify Git
Ensure Git is installed:
```bash
git --version
```

If not installed:

- **Windows**: [Download Git](https://git-scm.com/download/win)
- **macOS**: `brew install git` or `xcode-select --install`
- **Linux (Debian/Ubuntu)**: `sudo apt update && sudo apt install git`

### 3. Download the `.ns6` Data File

Uhh, find ur own blackrock data.

Save it to a `data/` folder inside the cloned repository.

---

## 💻 Platform-Specific Setup

### macOS

```bash
# Open Terminal
cd ~
git clone https://github.com/njnovo/spike_sorting.git
cd spike_sorting

# Create Conda environment
conda env create -f full_spikeinterface_environment_mac.yml
conda activate si_env

# Launch Jupyter
jupyter notebook
```

### Linux

```bash
# Open Terminal
cd ~
git clone https://github.com/njnovo/spike_sorting.git
cd spike_sorting

# Create Conda environment
conda env create -f full_spikeinterface_environment_linux_dandi.yml
conda activate si_env

# Launch Jupyter
jupyter notebook
```

### Windows (PowerShell Recommended)

```powershell
cd C:\Users\YourUsername\Documents
git clone https://github.com/njnovo/spike_sorting.git
cd spike_sorting

# Create Conda environment
conda env create -f full_spikeinterface_environment_windows.yml
conda activate si_env

# Launch Jupyter
jupyter notebook
```

---

## 📓 Running the Demos

1. Open `jupyter notebook` in the terminal.
2. Navigate to the `demos/` folder.
3. Open a demo file, e.g., `data_processing_demo.ipynb`.
4. Select kernel: **Kernel → Change Kernel → si_env**.
5. Update the data path:
   ```python
   FILE_PATH_TO_DATA = "data/Hub1-datafile001.ns6"
   ```
6. Run all cells: **Cell → Run All** or use `Shift + Enter`.

---

## 📬 Contact

For issues or questions, please reach out via GitHub or email the maintainer.

---
```
