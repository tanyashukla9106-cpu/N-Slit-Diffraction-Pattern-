"""
N-Slit Diffraction Pattern Simulation
-------------------------------------
- Asks for the number of slits, wavelength, slit width and slit separation.
- Shows the simulation immediately after the values are entered.
    * On a normal computer: an interactive matplotlib window opens.
    * In GitHub Codespaces (with no display): the plot is saved as a PNG and
      opened automatically in the editor.
- The program has been designed to ask whether you want to run another simulation.
- When you stop, the LAST simulation stays in the project folder as
  'last_simulation.png' (plus 'last_simulation_parameters.txt').
"""

import os
import shutil
import subprocess
import sys

import matplotlib

# Use a window if a display exists, otherwise draw to a file (Codespaces).
HAS_DISPLAY = (
    sys.platform in ("win32", "darwin")
    or bool(os.environ.get("DISPLAY"))
    or bool(os.environ.get("WAYLAND_DISPLAY"))
)
if not HAS_DISPLAY:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGE_FILE = os.path.join(FOLDER, "last_simulation.png")
PARAM_FILE = os.path.join(FOLDER, "last_simulation_parameters.txt")


def ask_number(prompt, minimum=0.0, integer=False):
    """Keep asking until a valid positive number is entered."""
    while True:
        text = input(prompt).strip()
        try:
            value = int(text) if integer else float(text)
        except ValueError:
            print("  Please enter a valid number.")
            continue
        if value < minimum or value == 0 and minimum == 0:
            print(f"  Value must be greater than {minimum:g}.")
            continue
        return value


def get_inputs():
    n = ask_number("Enter number of slits (n): ", minimum=1, integer=True)
    wavelength_nm = ask_number("Enter wavelength (in nm): ")
    while True:
        width_um = ask_number("Enter slit width (in micrometers): ")
        sep_um = ask_number("Enter distance between slits (in micrometers): ")
        if n > 1 and sep_um <= width_um:
            print("  Slit separation must be larger than slit width. Re-enter both.")
            continue
        break
    return n, wavelength_nm, width_um, sep_um


def compute_pattern(n, wavelength_nm, width_um, sep_um):
    lam = wavelength_nm * 1e-9
    a = width_um * 1e-6
    d = sep_um * 1e-6

    # Show about 3 lobes of the single-slit envelope (limited to +/-90 degrees)
    max_sin = min(3 * lam / a, 1.0)
    theta = np.linspace(-np.arcsin(max_sin), np.arcsin(max_sin), 15000)
    s = np.sin(theta)

    beta = np.pi * a * s / lam
    alpha = np.pi * d * s / lam

    envelope = np.sinc(beta / np.pi) ** 2            # single-slit diffraction

    denom = n * np.sin(alpha)
    ratio = np.ones_like(alpha)                      # limit value at principal maxima
    safe = np.abs(np.sin(alpha)) > 1e-9
    ratio[safe] = np.sin(n * alpha[safe]) / denom[safe]
    interference = ratio ** 2                        # N-slit interference

    intensity = envelope * interference
    intensity /= intensity.max()
    return np.degrees(theta), intensity, envelope


def build_figure(n, wavelength_nm, width_um, sep_um):
    theta_deg, intensity, envelope = compute_pattern(n, wavelength_nm, width_um, sep_um)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax1.plot(theta_deg, intensity, color="tab:blue", lw=1.3, label="Intensity")
    ax1.plot(theta_deg, envelope, "--", color="tab:red", lw=1, alpha=0.7,
             label="Single-slit envelope")
    ax1.set_ylabel("Normalized intensity")
    ax1.set_title(
        f"{n}-Slit Diffraction Pattern\n"
        f"λ = {wavelength_nm:g} nm, slit width = {width_um:g} µm, "
        f"separation = {sep_um:g} µm"
    )
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    # Screen view: how the fringes would look on a screen
    strip = np.tile(intensity, (60, 1))
    ax2.imshow(
        strip, aspect="auto", cmap="gray",
        extent=[theta_deg[0], theta_deg[-1], 0, 1],
    )
    ax2.set_yticks([])
    ax2.set_xlabel("Diffraction angle θ (degrees)")
    ax2.set_title("Screen view", fontsize=9)

    fig.tight_layout()
    return fig


def save_last_simulation(fig, params):
    fig.savefig(IMAGE_FILE, dpi=300)
    n, wl, w, s = params
    with open(PARAM_FILE, "w") as f:
        f.write("Last simulation parameters\n")
        f.write(f"Number of slits : {n}\n")
        f.write(f"Wavelength      : {wl:g} nm\n")
        f.write(f"Slit width      : {w:g} um\n")
        f.write(f"Slit separation : {s:g} um\n")


def open_in_editor(path):
    """In Codespaces / VS Code terminal, open the image in the editor."""
    code = shutil.which("code")
    if code:
        try:
            subprocess.run([code, path], check=False)
            return True
        except OSError:
            pass
    return False


def show_simulation(fig):
    # Every simulation is saved, so the last one is always kept in the folder.
    if HAS_DISPLAY:
        plt.show()          # interactive window; blocks until closed
    else:
        if not open_in_editor(IMAGE_FILE):
            print(f"  Open '{IMAGE_FILE}' from the Explorer to see the plot.")
        else:
            print("  Simulation opened in the editor.")


def main():
    print("N-Slit Diffraction Pattern Simulation")
    while True:
        params = get_inputs()
        fig = build_figure(*params)
        save_last_simulation(fig, params)   # saved BEFORE showing, so it is never lost
        show_simulation(fig)
        plt.close(fig)

        again = input("\nRun another simulation? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("\nDone. The last simulation is saved in the project folder:")
            print(f"  {IMAGE_FILE}")
            print(f"  {PARAM_FILE}")
            break
        print()


if __name__ == "__main__":
    main()