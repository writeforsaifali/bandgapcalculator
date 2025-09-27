"""
Streamlit application for UV-Vis spectroscopy data analysis.

This app provides an interactive user interface for analysing UV‑Vis
spectroscopy measurements stored in a CSV file. It replicates the
functionality of the original command line script, allowing users to
upload a dataset, inspect the raw data, compute absorption curves,
normalise and smooth those curves, calculate first and second
derivatives, identify the largest absorption edge peak for each
sample, and visualise the results through a series of plots. The app
also exposes a configuration sidebar for adjusting smoothing
parameters, selecting an energy conversion range and choosing which
samples to analyse.

When executed directly (for example by double‑clicking the file) the
script will launch a Streamlit server using the internal CLI. This
avoids the need to run ``streamlit run`` manually from the command
line. To use this feature you must have the ``streamlit`` package
installed in your Python environment.

Author: Saif Ali <sali@unisa.it>
"""

from __future__ import annotations

import io
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional plotting and signal-processing libraries. These are required
# for full functionality; on environments like Streamlit Cloud they must
# be present in requirements.txt. We import them with guarded
# try/except blocks so that the module can be loaded and we can show a
# helpful message to users instead of a ModuleNotFoundError traceback.
try:
    import matplotlib
    # Use a non-interactive backend suitable for headless environments
    try:
        matplotlib.use('Agg')
    except Exception:
        # If backend selection fails, continue — pyplot import may still work
        pass
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - environment dependent
    plt = None

try:
    from scipy.signal import savgol_filter, find_peaks
except Exception:  # pragma: no cover - environment dependent
    # Provide lightweight fallbacks to avoid NameError later. The
    # fallbacks are minimal and will raise at runtime if used; the
    # Streamlit UI will instruct users to install scipy instead.
    def savgol_filter(x, window_length, polyorder):
        raise RuntimeError('scipy is required for smoothing. Please install scipy.')

    def find_peaks(*args, **kwargs):
        raise RuntimeError('scipy is required for peak finding. Please install scipy.')

import streamlit as st

# Additional imports for file parsing
import csv

from typing import Callable, Any


def wavelength_to_energy(wavelength_nm: float) -> float:
    """Convert a wavelength in nanometres to photon energy in electron volts.

    Args:
        wavelength_nm: Wavelength in nanometres.

    Returns:
        The corresponding energy in electron volts.
    """
    return 1240.0 / wavelength_nm


def process_data(
    data: pd.DataFrame,
    samples: List[str],
    limit_rows: int,
    window_length: int = 55,
    polyorder: int = 2,
    derivative_range: Tuple[float, float] = (600.0, 900.0),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process the raw UV‑Vis data into a merged DataFrame with computed columns.

    This function extracts the wavelength column, calculates transmission,
    reflectance, absorption, normalised absorption, smoothed absorption,
    first and second derivatives for each sample, and finds the largest peak
    in the second derivative within a specified wavelength range.

    Args:
        data: A pandas DataFrame with a MultiIndex header. The top level
            contains the sample names and the second level contains the
            measurement type ("%T" or "%R"). The first column is assumed
            to be the wavelength.
        samples: List of sample names (top level of the MultiIndex) to
            include in the analysis.
        limit_rows: Number of rows from the top of ``data`` to process.
        window_length: Window length (must be odd) for the Savitzky–Golay
            smoothing filter.
        polyorder: Polynomial order for the Savitzky–Golay filter.
        derivative_range: Tuple specifying the inclusive wavelength range
            (min, max) over which to search for peaks in the second
            derivative.

    Returns:
        A tuple containing:
            * ``merged_data`` – DataFrame with calculated columns for
              wavelength, %T, %R, %A, normalised absorption, smoothed
              absorption and first/second derivatives for each sample.
            * ``peak_values`` – DataFrame summarising the largest peak
              location (wavelength) and corresponding energy for each
              sample.

    Notes:
        The Savitzky–Golay filter requires an odd window length and
        ``window_length > polyorder``. If these conditions are not met the
        window length will be adjusted to satisfy them.
    """
    # Ensure window_length is odd and greater than polyorder
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 3  # ensure it is odd later
        if window_length % 2 == 0:
            window_length += 1

    # Prepare merged_data with the wavelength column
    merged_data: pd.DataFrame = pd.DataFrame()
    # The first column in the CSV (top level) contains the wavelength; use
    # data.columns[0] rather than a hard‑coded name. Some CSV files
    # generated by spectrometers might have an unnamed level; this picks
    # whatever is present.
    first_col = data.columns[0]
    wavelength_series = data[first_col].iloc[:limit_rows].astype(float)
    merged_data['Wavelength (nm)'] = wavelength_series.reset_index(drop=True)

    peak_records: List[Dict[str, float]] = []
    # Process each selected sample. We attempt to compute absorption from
    # existing columns if available. Accepted second‑level labels for
    # absorption include '%A' and common variants like 'Abs' or 'Absorption'.
    for sample in samples:
        # Identify measurement keys
        t_key = (sample, '%T')
        r_key = (sample, '%R')
        # Try to locate an absorption column first
        abs_candidates = [
            (sample, '%A'),
            (sample, 'Abs'),
            (sample, 'Absorption'),
        ]
        abs_key = next((key for key in abs_candidates if key in data.columns), None)
        # Extract transmission and reflectance if present
        transmission = (
            data[t_key].iloc[:limit_rows].astype(float).reset_index(drop=True)
            if t_key in data.columns
            else None
        )
        reflectance = (
            data[r_key].iloc[:limit_rows].astype(float).reset_index(drop=True)
            if r_key in data.columns
            else None
        )
        # Determine absorption series
        if abs_key is not None:
            # Use provided absorption data directly
            absorption = data[abs_key].iloc[:limit_rows].astype(float).reset_index(drop=True)
            # If transmission/reflectance missing, compute them as zeros for storage
            if transmission is None:
                transmission = pd.Series([np.nan] * limit_rows)
            if reflectance is None:
                reflectance = pd.Series([np.nan] * limit_rows)
        else:
            # Compute absorption from transmission and reflectance when both are
            # available. If only transmission is available assume reflectance
            # negligible and derive absorption as 100 - transmission.
            if transmission is not None and reflectance is not None:
                absorption = 100.0 - (transmission + reflectance)
            elif transmission is not None:
                absorption = 100.0 - transmission
                # Set reflectance to NaN as it is unknown
                reflectance = pd.Series([np.nan] * limit_rows)
            else:
                # Insufficient information to compute absorption; skip sample
                continue
        # Normalise absorption between 0 and 1
        min_val = float(absorption.min())
        max_val = float(absorption.max())
        if max_val != min_val:
            normalised = (absorption - min_val) / (max_val - min_val)
        else:
            normalised = absorption * 0.0  # avoid division by zero
        # Smooth the normalised absorption
        smoothed = savgol_filter(normalised, window_length=window_length, polyorder=polyorder)
        # Compute derivatives
        wl = merged_data['Wavelength (nm)']
        first_derivative = np.gradient(smoothed, wl)
        second_derivative = np.gradient(np.gradient(smoothed, wl), wl)
        # Append columns to merged_data
        merged_data[f'{sample} %T'] = transmission
        merged_data[f'{sample} %R'] = reflectance
        merged_data[f'{sample} %A'] = absorption
        merged_data[f'{sample} %A Normalized'] = normalised
        merged_data[f'{sample} %A Normalized Smoothed'] = smoothed
        merged_data[f'{sample} %A Normalized 1st Derivative'] = first_derivative
        merged_data[f'{sample} %A Normalized 2nd Derivative'] = second_derivative
        # Find largest peak in second derivative within derivative_range
        xmin, xmax = derivative_range
        valid = (wl >= xmin) & (wl <= xmax)
        # Use peaks in the positive second derivative; peaks returns indices
        peaks, _ = find_peaks(second_derivative[valid])
        if peaks.size > 0:
            # Map relative indices back to absolute row indices in merged_data
            valid_indices = np.where(valid)[0]
            absolute_peaks = valid_indices[peaks]
            # Determine which peak has the maximum second derivative value
            largest_peak_idx = absolute_peaks[np.argmax(second_derivative[absolute_peaks])]
            peak_wl = float(wl.iloc[largest_peak_idx])
            energy = wavelength_to_energy(peak_wl)
            peak_records.append({'Sample': sample, 'Wavelength (nm)': peak_wl, 'Energy (eV)': energy})
        else:
            # No peak found in the range
            peak_records.append({'Sample': sample, 'Wavelength (nm)': float('nan'), 'Energy (eV)': float('nan')})

    # Assemble the peak values DataFrame
    peak_values = pd.DataFrame(peak_records)
    return merged_data, peak_values


def plot_lines(
    merged_data: pd.DataFrame,
    samples: List[str],
    column_suffix: str,
    y_label: str,
    title: str,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    annotate_peaks: bool = False,
    peak_values: pd.DataFrame | None = None,
    *,
    cmap_name: str = 'copper',
    line_style: str = '-',
    line_width: float = 1.5,
    marker: Optional[str] = None,
    legend_loc: str = 'best',
    show_grid: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 150,
) -> Any:
    """Create a multi‑line plot for a given measurement across samples.

    This function is an extended version of the original ``plot_lines``
    function. In addition to plotting the specified measurement for
    each sample, it allows users to customise the appearance of the
    resulting figure. Users can specify a colour map, line style,
    line width, marker style, legend location, grid visibility, figure
    size and DPI (resolution).

    Args:
        merged_data: Processed data containing wavelength and measurement
            columns.
        samples: List of sample identifiers to plot.
        column_suffix: Suffix appended to each sample name to form the
            column name (e.g. '%T', '%A Normalized 2nd Derivative').
        y_label: Axis label for the y‑axis.
        title: Title of the plot.
        xlim: Optional tuple specifying x‑axis limits.
        ylim: Optional tuple specifying y‑axis limits.
        annotate_peaks: Whether to annotate peak energy values on the
            plot. If ``True`` and ``peak_values`` is provided, vertical
            lines and labels will be added at the detected peak
            wavelengths.
        peak_values: DataFrame of peak information used for annotations.
        cmap_name: Name of the matplotlib colormap used to assign colours
            to the sample lines.
        line_style: Matplotlib line style (e.g. '-', '--', '-.', ':').
        line_width: Width of the plotted lines.
        marker: Marker style for data points (e.g. 'o', 's', '^'). Use
            ``None`` or the string 'None' to disable markers.
        legend_loc: Location of the legend (any valid matplotlib
            location string).
        show_grid: Whether to display grid lines on the plot.
        figsize: Tuple specifying the width and height of the figure in
            inches.
        dpi: Resolution of the figure in dots per inch.

    Returns:
        A matplotlib ``Figure`` object ready for display in Streamlit
        and for saving as a high‑resolution image.
    """
    # Create the figure and axis with the requested size and resolution
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Obtain the colormap and compute distinct colours
    try:
        cmap = plt.get_cmap(cmap_name)
    except Exception:
        # Fallback to a default colormap if the specified one is invalid
        cmap = plt.get_cmap('viridis')
    colours = cmap(np.linspace(0, 1, max(len(samples), 1)))
    # Normalise marker: treat the string 'None' as no marker
    marker_style = None if (marker is None or str(marker).lower() == 'none') else marker
    for idx, sample in enumerate(samples):
        col_name = f'{sample} {column_suffix}'
        if col_name not in merged_data.columns:
            continue
        ax.plot(
            merged_data['Wavelength (nm)'],
            merged_data[col_name],
            label=sample,
            color=colours[idx % len(colours)],
            linestyle=line_style,
            linewidth=line_width,
            marker=marker_style,
        )
    # Set axis labels and title
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    # Axis limits
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # Grid
    ax.grid(show_grid)
    # Annotate peaks if requested
    if annotate_peaks and peak_values is not None:
        for _, row in peak_values.iterrows():
            sample = row['Sample']
            peak_wl = row['Wavelength (nm)']
            energy = row['Energy (eV)']
            if not np.isnan(peak_wl):
                ax.axvline(x=peak_wl, color='grey', linestyle='--', linewidth=0.8)
                # Position annotation text at 90 % of the y‑axis maximum
                y_pos = ax.get_ylim()[1] * 0.9
                ax.text(
                    peak_wl,
                    y_pos,
                    f'{energy:.2f} eV',
                    rotation=90,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
                )
    # Legend
    ax.legend(loc=legend_loc)
    return fig


# -----------------------------------------------------------------------------
# Data loading and merging utilities
# -----------------------------------------------------------------------------

def flatten_multiindex_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a DataFrame with a MultiIndex column into a single-level column.

    The wavelength column is renamed to ``'Wavelength (nm)'``. All other
    columns are joined with a space between the sample name and measurement.

    Args:
        df: A pandas DataFrame with MultiIndex columns. One of the columns
            should be ``('global', 'Wavelength (nm)')`` representing the
            wavelength.

    Returns:
        A DataFrame with single-level column names suitable for merging.
    """
    new_cols: List[str] = []
    for c in df.columns:
        if isinstance(c, tuple) and len(c) == 2:
            sample, meas = c
            if sample == 'global' and meas == 'Wavelength (nm)':
                new_cols.append('Wavelength (nm)')
            else:
                # Join sample and measurement with a space. Strip stray spaces.
                new_cols.append(f'{sample} {meas}'.strip())
        else:
            # Fallback: use the existing column name
            new_cols.append(str(c))
    df_flat = df.copy()
    df_flat.columns = new_cols
    return df_flat


def multiindex_from_flat(df_flat: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a MultiIndex DataFrame from a flattened DataFrame.

    Column names must follow the format ``'sample measurement'`` where
    ``'sample'`` can contain spaces. The last token is interpreted as the
    measurement. The column ``'Wavelength (nm)'`` is mapped back to
    ``('global', 'Wavelength (nm)')``.

    Args:
        df_flat: A DataFrame with single-level column names.

    Returns:
        A DataFrame with a two-level MultiIndex for columns.
    """
    new_cols: List[Tuple[str, str]] = []
    for c in df_flat.columns:
        if c == 'Wavelength (nm)':
            new_cols.append(('global', 'Wavelength (nm)'))
        else:
            parts = str(c).split()
            # The measurement is the last token; the sample name is the rest
            if len(parts) >= 2:
                measurement = parts[-1]
                sample = ' '.join(parts[:-1])
            else:
                sample = parts[0]
                measurement = ''
            new_cols.append((sample, measurement))
    df_multi = df_flat.copy()
    df_multi.columns = pd.MultiIndex.from_tuples(new_cols)
    return df_multi


def parse_two_row_header_content(content: str) -> Optional[pd.DataFrame]:
    """Parse a CSV-like text where the first two lines are header rows.

    The first line contains sample names and the second line contains
    measurement identifiers. Subsequent rows contain numeric data. Blank
    entries in the first line inherit the previous non-blank sample name.

    Args:
        content: The entire file content as a string.

    Returns:
        A DataFrame with a two-level MultiIndex on the columns if parsing
        succeeds, or ``None`` on failure.
    """
    lines = content.splitlines()
    if len(lines) < 3:
        return None
    try:
        reader = list(csv.reader(lines))
    except Exception:
        return None
    header1 = reader[0]
    header2 = reader[1]
    # Build column MultiIndex by propagating sample names
    sample = None
    columns: List[Tuple[str, str]] = []
    for h1, h2 in zip(header1, header2):
        h1 = (h1 or '').strip()
        h2 = (h2 or '').strip()
        if h1:
            sample = h1
        columns.append((sample, h2))
    # If the last measurement entry is blank, drop it
    # Identify rows with numeric data
    data_rows = reader[2:]
    df = pd.DataFrame(data_rows, columns=columns)
    # Convert numeric values where possible
    for col in df.columns:
        # Leave wavelength column as string; numeric conversion later
        if col[1] != 'Wavelength (nm)':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col], errors='ignore')
    # Rename the first occurrence of 'Wavelength (nm)' to global and drop
    # duplicate wavelength columns
    global_col: Optional[Tuple[str, str]] = None
    for col in df.columns:
        if col[1] == 'Wavelength (nm)':
            global_col = col
            break
    if global_col is not None:
        df = df.rename(columns={global_col: ('global', 'Wavelength (nm)')})
    # Drop other wavelength columns
    other_wl = [c for c in df.columns if c[1] == 'Wavelength (nm)' and c[0] != 'global']
    if other_wl:
        df = df.drop(columns=other_wl)
    # Drop columns with empty measurement names
    df = df.drop(columns=[c for c in df.columns if not c[1]])
    return df


def parse_simple_header_df(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Parse a DataFrame with a single header row into a MultiIndex DataFrame.

    This function attempts to identify the wavelength column and measurement
    columns based on column names. Measurement columns are assumed to be
    named in the pattern ``<sample> <measurement>`` (e.g. ``PVK1 %T``).
    If column names do not follow this pattern, the file name (without
    extension) is used as the sample name and the column name as the
    measurement. The wavelength column is identified by searching for
    substrings like ``'wave'`` in the column name or by taking the first
    column.

    Args:
        df: DataFrame loaded from the file.
        file_name: Name of the file without the extension, used as a
            default sample name.

    Returns:
        A DataFrame with a two-level MultiIndex on the columns.
    """
    # Identify wavelength column
    wavelength_col = None
    for col in df.columns:
        if isinstance(col, str) and 'wave' in col.lower():
            wavelength_col = col
            break
    if wavelength_col is None:
        # Fallback: use the first column
        wavelength_col = df.columns[0]
    # Build lists of tuples for multi-index
    columns: List[Tuple[str, str]] = []
    for col in df.columns:
        if col == wavelength_col:
            columns.append(('global', 'Wavelength (nm)'))
        else:
            # Try to split sample and measurement
            if isinstance(col, str) and ' ' in col:
                parts = col.split()
                measurement = parts[-1]
                sample = ' '.join(parts[:-1])
            else:
                sample = file_name
                measurement = str(col)
            columns.append((sample, measurement))
    df_multi = df.copy()
    df_multi.columns = pd.MultiIndex.from_tuples(columns)
    # Ensure wavelength is numeric for proper merging
    df_multi[('global', 'Wavelength (nm)')] = pd.to_numeric(
        df_multi[('global', 'Wavelength (nm)')], errors='ignore'
    )
    return df_multi


def read_uvvis_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Read a single uploaded file and return a MultiIndex DataFrame.

    The function detects the file type by its extension and applies the
    appropriate parsing logic. CSV and Excel files may contain either
    multi-index headers or two-row headers (sample names and measurement
    identifiers). Text files are assumed to contain two columns (wavelength
    and transmittance) and the sample name is derived from the file name.

    Args:
        uploaded_file: A Streamlit UploadedFile object representing the
            uploaded file.

    Returns:
        A DataFrame with a two-level MultiIndex on the columns, or ``None``
        if the file could not be parsed.
    """
    import os
    name = uploaded_file.name
    base, ext = os.path.splitext(name)
    ext = ext.lower()
    # Read the file content into memory. Attempt multiple encodings to
    # gracefully handle text files saved with various code pages (e.g.
    # UTF‑8, UTF‑16, ANSI/Windows encodings). We decode the bytes into
    # a Unicode string, preferring UTF‑8 and falling back to other
    # encodings. If all attempts fail we resort to ignoring invalid
    # characters.
    try:
        content_bytes = uploaded_file.getvalue()
    except Exception:
        return None
    # List of candidate encodings to try. The order reflects common
    # encodings encountered in spectrometer exports.
    candidate_encodings = ['utf-8', 'utf-16', 'cp1252', 'latin-1']
    content_str = None
    for enc in candidate_encodings:
        try:
            content_str = content_bytes.decode(enc)
            break
        except Exception:
            continue
    if content_str is None:
        # Fallback: decode ignoring errors
        content_str = content_bytes.decode('utf-8', errors='ignore')
    if ext in ['.txt']:
        # Attempt to parse plain text files exported from spectrometers.
        # First, try to handle files that include two header rows: a sample
        # name followed by measurement identifiers (e.g. "Wavelength nm.", "T%", "R%", etc.).
        try:
            # Use csv.reader to correctly handle quoted fields and delimiters
            rows = list(csv.reader(content_str.splitlines()))
            if len(rows) >= 3:
                header1 = [h.strip().strip('"') for h in rows[0]]
                header2 = [h.strip().strip('"') for h in rows[1]]
                # If the first header row has only one value (the sample name)
                # and the second header row contains multiple measurement names,
                # construct a MultiIndex DataFrame accordingly.
                if len(header1) == 1 and len(header2) >= 2:
                    sample_name = header1[0] if header1[0] else base
                    # Build list of MultiIndex columns
                    columns: List[Tuple[str, str]] = []
                    for idx, meas_name in enumerate(header2):
                        # Standardise measurement identifiers
                        name_lower = meas_name.lower().replace('%', '').replace(' ', '')
                        if 'wave' in name_lower or 'nm' in name_lower:
                            columns.append(('global', 'Wavelength (nm)'))
                        else:
                            # Map measurement type to %T, %R, %A if possible
                            if 't' in name_lower:
                                measurement = '%T'
                            elif 'r' in name_lower:
                                measurement = '%R'
                            elif 'a' in name_lower:
                                measurement = '%A'
                            else:
                                # Fallback: use raw name
                                measurement = meas_name
                            columns.append((sample_name, measurement))
                    # Read data rows starting after the two header lines
                    data_rows = rows[2:]
                    # Only retain rows with the correct number of columns
                    data_rows = [r for r in data_rows if len(r) >= len(columns)]
                    if data_rows:
                        df = pd.DataFrame(data_rows, columns=columns)
                        # Convert numeric columns
                        for col in df.columns:
                            if col[1] == 'Wavelength (nm)':
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            else:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        return df
        except Exception:
            pass
        # Fallback: parse as a simple two‑column dataset (wavelength, transmission)
        try:
            # Split into lines and attempt to skip header lines until we
            # encounter a row where the first value can be interpreted as a
            # number. This avoids mistaking sample names for numeric data.
            lines = content_str.splitlines()
            skiprows = 0
            for line in lines:
                stripped = line.strip().strip('"')
                # Split by common delimiters to inspect the first token
                if ',' in stripped:
                    first_token = stripped.split(',')[0]
                elif '\t' in stripped:
                    first_token = stripped.split('\t')[0]
                else:
                    # Use whitespace
                    first_token = stripped.split()[0] if stripped.split() else ''
                # Check if the first token is numeric
                try:
                    float(first_token)
                    break
                except Exception:
                    skiprows += 1
                    continue
            # Determine delimiter by inspecting the first numeric row
            delim = ','
            for line in lines[skiprows:]:
                if ',' in line:
                    delim = ','
                    break
                elif '\t' in line:
                    delim = '\t'
                    break
                elif ' ' in line:
                    delim = '\s+'
                    break
            import io as _io
            f = _io.StringIO('\n'.join(lines[skiprows:]))
            if delim == '\s+':
                df = pd.read_csv(f, delim_whitespace=True, header=None)
            else:
                df = pd.read_csv(f, sep=delim, header=None)
            # If the first row contains non-numeric text, treat it as a header
            if df.shape[1] >= 2:
                try:
                    _ = float(str(df.iloc[0, 0]).replace(',', '.'))
                    header_present = False
                except Exception:
                    header_present = True
                if header_present:
                    df.columns = df.iloc[0]
                    df = df.drop(df.index[0])
            # Only keep the first two columns: wavelength and transmittance
            if df.shape[1] < 2:
                return None
            df = df.iloc[:, :2]
            df.columns = ['Wavelength (nm)', f'{base} %T']
            # Convert values to numeric
            df['Wavelength (nm)'] = pd.to_numeric(df['Wavelength (nm)'], errors='coerce')
            df[f'{base} %T'] = pd.to_numeric(df[f'{base} %T'], errors='coerce')
            # Build MultiIndex DataFrame
            new_cols = [('global', 'Wavelength (nm)'), (base, '%T')]
            df_multi = df.copy()
            df_multi.columns = pd.MultiIndex.from_tuples(new_cols)
            return df_multi
        except Exception:
            return None
    else:
        # Handle CSV and Excel
        # First attempt: parse as two-row header
        try:
            df_two = parse_two_row_header_content(content_str)
            if df_two is not None:
                return df_two
        except Exception:
            pass
        # Second attempt: read with pandas using header=[0,1]
        try:
            if ext in ['.xls', '.xlsx']:
                df_mi = pd.read_excel(uploaded_file, header=[0, 1])
            else:
                # Reset pointer for pandas to read
                uploaded_file.seek(0)
                df_mi = pd.read_csv(uploaded_file, header=[0, 1])
            # Check if we have multi-index columns with a reasonable structure
            if isinstance(df_mi.columns, pd.MultiIndex) and df_mi.columns.nlevels == 2:
                # Rename first wavelength column to global and drop duplicates
                df_mi = df_mi.copy()
                # Find the first wavelength column
                global_col = None
                for col in df_mi.columns:
                    if col[1] == 'Wavelength (nm)':
                        global_col = col
                        break
                if global_col is not None:
                    df_mi = df_mi.rename(columns={global_col: ('global', 'Wavelength (nm)')})
                # Drop other wavelength columns
                other = [c for c in df_mi.columns if c[1] == 'Wavelength (nm)' and c[0] != 'global']
                if other:
                    df_mi = df_mi.drop(columns=other)
                # Drop empty measurement columns
                df_mi = df_mi.drop(columns=[c for c in df_mi.columns if not c[1]])
                return df_mi
        except Exception:
            pass
        # Third attempt: simple header parsing
        try:
            # Reset pointer
            uploaded_file.seek(0)
            if ext in ['.xls', '.xlsx']:
                df_simple = pd.read_excel(uploaded_file, header=0)
            else:
                df_simple = pd.read_csv(uploaded_file, header=0)
            df_mi = parse_simple_header_df(df_simple, base)
            return df_mi
        except Exception:
            pass
    return None


def merge_dataframes(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Merge multiple MultiIndex DataFrames on the wavelength column.

    Args:
        dfs: List of MultiIndex DataFrames. Each DataFrame must include a
            column ``('global', 'Wavelength (nm)')``.

    Returns:
        A single MultiIndex DataFrame containing all columns from the input
        DataFrames merged on the wavelength column. If the list is empty,
        returns ``None``.
    """
    if not dfs:
        return None
    # Start with the first DataFrame
    combined = dfs[0]
    for df in dfs[1:]:
        # Flatten both DataFrames to single-level columns
        combined_flat = flatten_multiindex_df(combined)
        df_flat = flatten_multiindex_df(df)
        # Merge on the wavelength column
        merged_flat = pd.merge(
            combined_flat,
            df_flat,
            on='Wavelength (nm)',
            how='outer',
        )
        # Convert back to MultiIndex
        combined = multiindex_from_flat(merged_flat)
    return combined


def compute_tauc_bandgap(
    merged_data: pd.DataFrame,
    samples: List[str],
    limit_rows: int,
    tauc_exponent: float = 2.0,
    energy_range: Optional[Tuple[float, float]] = None,
    smooth_window: int = 21,
    smooth_polyorder: int = 2,
    threshold_fraction: float = 0.1,
    thicknesses: Optional[Dict[str, float]] = None,
    enable_fallback: bool = True,
    show_diagnostics: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Tuple[float, float]], Dict[str, pd.DataFrame]]:
    """
    Compute band gaps for multiple samples using a flexible Tauc plot approach.

    This function supports two pathways for constructing Tauc plots depending on the
    availability of optical data:

    1. **Reflectance‑based**: When a sample has reflectance (%R) data, the
       Kubelka–Munk function ``F(R) = (1−R)^2 / (2R)`` is computed and used to
       construct the Tauc quantity ``(F(R) * E)^n``, where ``E`` is photon
       energy and ``n`` is the Tauc exponent.
    2. **Transmittance‑based**: If reflectance is unavailable but a sample has
       transmittance (%T) data and a positive film thickness is provided, the
       absorption coefficient is estimated via ``alpha = −ln(T) / d`` (with ``d``
       in metres). The Tauc quantity is then ``(alpha * E)^n``. If thickness
       is missing or non‑positive, such samples are skipped.

    For each sample, the smoothed Tauc curve is restricted to a specified
    energy range and a linear region above a threshold fraction of its maximum
    is selected for fitting. A straight line is fitted via ``numpy.polyfit``,
    and the band gap ``E_g`` is taken as the x‑intercept ``−intercept/slope`` if
    the slope is positive. Invalid or insufficient data lead to ``NaN`` values.

    Parameters
    ----------
    merged_data : pandas.DataFrame
        Wide‑format DataFrame containing a column ``('global', 'Wavelength (nm)')``
        and, for each sample, optional columns ``(sample, '%R')`` and
        ``(sample, '%T')`` with reflectance and transmittance percentages.
    samples : list of str
        Names of samples to process. Samples lacking both ``%R`` and ``%T``
        columns will be skipped.
    limit_rows : int
        Number of initial rows to use from ``merged_data``. Subsequent rows
        are ignored. This matches the behaviour of the original UV‑Vis analysis.
    tauc_exponent : float, optional
        Exponent ``n`` in the Tauc relation ``(X * E)^n``, where ``X`` is
        either ``F(R)`` or the absorption coefficient ``alpha``. Typical
        values are 2.0 for direct allowed transitions and 0.5 for indirect.
    energy_range : tuple of float, optional
        Lower and upper bounds (in eV) of the photon energy region used for
        linear fitting.
    smooth_window : int, optional
        Window length (odd integer) for Savitzky–Golay smoothing of the
        computed Tauc curves.
    smooth_polyorder : int, optional
        Polynomial order for Savitzky–Golay smoothing. Must be less than
        ``smooth_window``.
    threshold_fraction : float, optional
        Fraction of the maximum smoothed Tauc value that defines the linear
        fitting region. Data points with ``Tauc >= threshold_fraction * max(Tauc)``
        are considered. If too few points remain, the top half of the
        restricted range is used as a fallback.
    thicknesses : dict of {str: float} or None, optional
        Mapping from sample names to film thicknesses in nanometres. Required
        when reflectance is absent and transmittance must be used to compute the
        absorption coefficient for that sample.  If a sample is not present
        in the dictionary or its thickness is non‑positive, that sample will
        be skipped when reflectance is absent.  If ``None``, all samples
        lacking reflectance will be skipped.

    Returns
    -------
    bandgap_df : pandas.DataFrame
        Table summarising the extracted band gaps for each sample. Contains
        columns ``Sample`` and ``Eg (eV)``. Samples that could not be
        processed have ``NaN`` band gaps.
    tauc_curves : dict
        Mapping from sample name to a DataFrame with columns ``E (eV)`` and
        ``Tauc`` representing the smoothed Tauc curve used for plotting.
    fit_params : dict
        Mapping from sample name to a tuple ``(slope, intercept)`` of the
        fitted line. Values are ``(np.nan, np.nan)`` when a fit is not
        possible or the slope is non‑positive.

    Notes
    -----
    - The photon energy is calculated as ``E = 1240 / wavelength`` (eV), with
      wavelength in nanometres.
    - Smoothing helps suppress noise but can broaden features; adjust
      ``smooth_window`` and ``smooth_polyorder`` to suit your data.
    - For transmittance‑based Tauc plots, the absorption coefficient is
      approximated as ``alpha = −ln(T)/d``. This assumes negligible
      reflectance and scattering losses and may not be accurate for all
      materials.
    """
    # Ensure smoothing window is odd and greater than polyorder
    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window <= smooth_polyorder:
        smooth_window = smooth_polyorder + 3
        if smooth_window % 2 == 0:
            smooth_window += 1
    # Prepare results containers
    bandgap_records: List[Dict[str, Any]] = []
    tauc_curves: Dict[str, pd.DataFrame] = {}
    fit_params: Dict[str, Tuple[float, float]] = {}
    # Detailed data per sample (raw measurements, computed alpha/F(R), energy, Tauc)
    detailed_data: Dict[str, pd.DataFrame] = {}
    # Extract the global wavelength column
    wl_col = ('global', 'Wavelength (nm)')
    if wl_col not in merged_data.columns:
        # If wavelength column is missing, return empty results for all outputs
        return (
            pd.DataFrame(columns=['Sample', 'Eg (eV)']),
            {},  # empty Tauc curves
            {},  # empty fit parameters
            {},  # empty detailed data
        )
    wavelengths = merged_data[wl_col].iloc[:limit_rows].astype(float)
    # Compute photon energy (eV)
    energies = 1240.0 / wavelengths
    # Process each sample
    for sample in samples:
        # Attempt to locate reflectance or transmittance columns for this sample
        # Candidate keys for reflectance
        refl_keys = ['%R', 'R%', 'R', 'Reflectance', 'reflectance', 'Reflectance (%)', 'Reflectance %']
        trans_keys = [
            '%T', 'T%', 'T', 't%', 't',
            'Transmittance', 'transmittance',
            'Transmittance (%)', 'Transmittance %',
            'Transmission', 'Transmission (%)', 'Transmission %', 'transmission', 'transmission (%)',
            'Trans', 'trans',
        ]
        r_col = None
        for rk in refl_keys:
            candidate = (sample, rk)
            if candidate in merged_data.columns:
                r_col = candidate
                break
        t_col = None
        for tk in trans_keys:
            candidate = (sample, tk)
            if candidate in merged_data.columns:
                t_col = candidate
                break
        base_values = None  # Will hold either F(R) or alpha
        # Case 1: Use reflectance if available
        if r_col is not None:
            # Get reflectance values as a fraction
            R = merged_data[r_col].iloc[:limit_rows].astype(float) / 100.0
            # Replace non‑positive values with NaN to avoid division errors
            R_safe = R.copy()
            R_safe[R_safe <= 0] = np.nan
            # Compute Kubelka–Munk function F(R) = (1 − R)^2 / (2R)
            base_values = ((1.0 - R_safe) ** 2) / (2.0 * R_safe)
        # Case 2: Use transmittance if reflectance is absent and per‑sample thickness provided
        elif t_col is not None:
            # Use thickness for this sample if provided
            if thicknesses is None:
                base_values = None
            else:
                thickness_nm = thicknesses.get(sample)
                if thickness_nm is not None and thickness_nm > 0:
                    # Get transmittance values as a fraction
                    T = merged_data[t_col].iloc[:limit_rows].astype(float) / 100.0
                    # Replace non‑positive values with NaN to avoid log errors
                    T_safe = T.copy()
                    T_safe[T_safe <= 0] = np.nan
                    # Convert thickness from nm to meters
                    d_m = thickness_nm * 1e-9
                    # Compute absorption coefficient alpha = −ln(T) / d
                    with np.errstate(divide='ignore', invalid='ignore'):
                        base_values = -np.log(T_safe) / d_m
                else:
                    base_values = None
        else:
            # Insufficient data for this sample: record NaN and skip
            bandgap_records.append({'Sample': sample, 'Eg (eV)': np.nan})
            fit_params[sample] = (np.nan, np.nan)
            continue
        # If base_values is None or entirely NaN, skip sample
        if base_values is None or base_values.isna().all():
            bandgap_records.append({'Sample': sample, 'Eg (eV)': np.nan})
            fit_params[sample] = (np.nan, np.nan)
            detailed_data[sample] = pd.DataFrame()
            continue
        # Compute Tauc y‑values (unsmoothed): (base * E)^n
        tauc_y = (base_values * energies) ** tauc_exponent
        # Build detailed data table for this sample
        try:
            if r_col is not None:
                raw_measure = merged_data[r_col].iloc[:limit_rows].astype(float)
                measure_name = r_col[1]
                measure_fraction = R  # reflectance fraction
                # Use F(R) as base
                f_r = base_values
                alpha_vals = None
            else:
                raw_measure = merged_data[t_col].iloc[:limit_rows].astype(float)
                measure_name = t_col[1]
                measure_fraction = merged_data[t_col].iloc[:limit_rows].astype(float) / 100.0
                # compute alpha = -ln(T)/d_m if thickness used; base_values already holds alpha
                f_r = None
                alpha_vals = base_values
            detailed_df = pd.DataFrame({
                'Wavelength (nm)': wavelengths,
                'Measurement': raw_measure,
                'Measurement type': measure_name,
                'Measurement fraction': measure_fraction,
                'F(R)': f_r,
                'Alpha': alpha_vals,
                'Energy (eV)': energies,
                'Tauc_raw': tauc_y,
            })
        except Exception:
            detailed_df = pd.DataFrame({
                'Wavelength (nm)': wavelengths,
                'Energy (eV)': energies,
                'Tauc_raw': tauc_y,
            })
        # Combine energies and tauc values into a DataFrame
        tauc_df = pd.DataFrame({'E (eV)': energies, 'Tauc': tauc_y})
        # Drop rows with invalid values
        tauc_df = tauc_df.dropna().reset_index(drop=True)
        if tauc_df.empty:
            bandgap_records.append({'Sample': sample, 'Eg (eV)': np.nan})
            fit_params[sample] = (np.nan, np.nan)
            detailed_data[sample] = detailed_df
            continue
        # Apply Savitzky–Golay smoothing to Tauc values
        try:
            smoothed = savgol_filter(
                tauc_df['Tauc'].values,
                window_length=smooth_window,
                polyorder=smooth_polyorder,
            )
        except Exception:
            smoothed = tauc_df['Tauc'].values
        tauc_df['Tauc_smooth'] = smoothed
        # Determine energy range for fitting.  If a specific range is provided, use it;
        # otherwise, use the full range of energies.
        if energy_range is not None:
            e_min, e_max = energy_range
        else:
            e_min = float(np.nanmin(energies))
            e_max = float(np.nanmax(energies))
        mask_range = (tauc_df['E (eV)'] >= e_min) & (tauc_df['E (eV)'] <= e_max)
        df_range = tauc_df.loc[mask_range].copy().reset_index(drop=True)
        if df_range.empty:
            bandgap_records.append({'Sample': sample, 'Eg (eV)': np.nan})
            fit_params[sample] = (np.nan, np.nan)
            continue
        # Determine threshold based on smoothed Tauc values and attempt fitting.
        max_val = df_range['Tauc_smooth'].max()
        if not np.isfinite(max_val) or max_val <= 0:
            bandgap_records.append({'Sample': sample, 'Eg (eV)': np.nan})
            fit_params[sample] = (np.nan, np.nan)
            continue

        # We'll attempt multiple threshold fractions as fallbacks when the
        # initial fit fails or yields a non-positive slope.
        tried_thresholds = []
        slopes = np.array([])
        intercepts = np.array([])
        Eg = np.nan
        slope = np.nan
        intercept = np.nan
        # Create a list of candidate thresholds (start with provided one)
        candidate_thresholds = [threshold_fraction, threshold_fraction / 2.0, threshold_fraction / 4.0, 0.01, 0.0]
        for thr in candidate_thresholds:
            if thr in tried_thresholds:
                continue
            tried_thresholds.append(thr)
            threshold = thr * max_val
            mask_threshold = df_range['Tauc_smooth'] >= threshold
            df_fit = df_range.loc[mask_threshold]
            # Fallback: if too few points, take top half of the energy‑range data
            if df_fit.shape[0] < 2:
                df_fit = df_range.sort_values('Tauc_smooth', ascending=False).iloc[: max(2, df_range.shape[0] // 2)]
            try:
                x_vals = df_fit['E (eV)'].values
                y_vals = df_fit['Tauc_smooth'].values
                coeffs = np.polyfit(x_vals, y_vals, 1)
                slope_try, intercept_try = coeffs[0], coeffs[1]
                slopes = np.append(slopes, slope_try)
                intercepts = np.append(intercepts, intercept_try)
                # Accept this fit if slope is positive
                if np.isfinite(slope_try) and slope_try > 0:
                    slope = slope_try
                    intercept = intercept_try
                    Eg = -intercept / slope if slope != 0 else np.nan
                    used_threshold = thr
                    used_df_fit = df_fit
                    break
                else:
                    # continue trying other thresholds
                    used_df_fit = df_fit
                    used_threshold = thr
            except Exception:
                used_df_fit = df_fit
                used_threshold = thr
                continue
        # If no positive-slope fit was found but we have any fit values, pick
        # the one with the largest positive slope (if any) else leave NaN.
        if not np.isfinite(slope) and slopes.size > 0:
            pos_idx = np.where(slopes > 0)[0]
            if pos_idx.size > 0:
                idx = pos_idx[np.argmax(slopes[pos_idx])]
                slope = slopes[idx]
                intercept = intercepts[idx]
                Eg = -intercept / slope if slope != 0 else np.nan
                # used_df_fit remains last tried; that's acceptable for diagnostics
        # Store diagnostics into detailed_df by marking which rows were used
        try:
            detailed_df['Fit used'] = detailed_df['Energy (eV)'].isin(used_df_fit['E (eV)'])
            detailed_df['Fit threshold used'] = float(used_threshold)
        except Exception:
            # If anything fails, create default columns
            detailed_df['Fit used'] = False
            detailed_df['Fit threshold used'] = float(threshold_fraction)
    bandgap_records.append({'Sample': sample, 'Eg (eV)': Eg})
    fit_params[sample] = (slope, intercept)
    # Prepare smoothed Tauc curve for plotting
    tauc_plot_df = df_range[['E (eV)', 'Tauc_smooth']].rename(columns={'Tauc_smooth': 'Tauc'})
    tauc_curves[sample] = tauc_plot_df
    # Store detailed data
    detailed_data[sample] = detailed_df
    bandgap_df = pd.DataFrame(bandgap_records)
    return bandgap_df, tauc_curves, fit_params, detailed_data


def compute_tauc_alpha_bandgap(
    merged_data: pd.DataFrame,
    samples: List[str],
    limit_rows: int,
    exponent: float = 2.0,
    thicknesses: Optional[Dict[str, float]] = None,
    glass_sample: Optional[str] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    smooth_window: int = 21,
    smooth_polyorder: int = 2,
    threshold_fraction: float = 0.1,
    enable_fallback: bool = True,
    show_diagnostics: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Tuple[float, float]], Dict[str, pd.DataFrame]]:
    """
    Compute band gaps for multiple samples using the absorption‑based Tauc plot.

    This function constructs Tauc plots from transmittance data by first
    correcting the sample transmittance with an optional glass/substrate
    measurement, converting the corrected transmittance to an absorption
    coefficient via Beer–Lambert's law, and then plotting ``(αE)^n`` versus
    photon energy ``E = 1240/λ``. A linear region of the resulting curve is
    identified and fitted to extract the optical band gap ``E_g``.  Smoothing
    and thresholding parameters allow the user to tailor the fit to their
    specific data.

    Parameters
    ----------
    merged_data : pandas.DataFrame
        Wide‑format DataFrame containing a column ``('global', 'Wavelength (nm)')``
        and, for each sample, a transmittance column ``(sample, '%T')`` in
        percentage units.
    samples : list of str
        Names of samples to process.  Samples lacking a ``%T`` column will be
        skipped.
    limit_rows : int
        Number of initial rows to use from ``merged_data``.
    exponent : float, optional
        Exponent ``n`` in the relation ``(αE)^n``.  Typical values are 2 for
        direct allowed transitions and 0.5 for indirect allowed transitions.
    thicknesses : dict of {str: float}, optional
        Mapping from sample names to film thicknesses in nanometres.  If a
        sample is absent from the dictionary or its thickness is non‑positive,
        that sample will be skipped because the absorption coefficient cannot
        be computed.  When ``None``, all samples will be skipped.
    glass_sample : str or None, optional
        Name of a sample whose transmittance should be used to correct the
        other samples.  If provided, its ``%T`` column is divided into the
        sample transmittance to remove the substrate contribution.  The glass
        sample itself is not analysed.  If ``None``, no correction is
        performed.
    energy_range : tuple of float or None, optional
        Lower and upper bounds (in eV) of the photon energy region used for
        linear fitting.  If ``None``, the full energy range derived from
        the wavelength data will be used.
    smooth_window : int, optional
        Window length (odd integer) for Savitzky–Golay smoothing of the
        computed Tauc curves.
    smooth_polyorder : int, optional
        Polynomial order for Savitzky–Golay smoothing.  Must be less than
        ``smooth_window``.
    threshold_fraction : float, optional
        Fraction of the maximum smoothed Tauc value that defines the linear
        fitting region.  Data points with ``Tauc >= threshold_fraction * max(Tauc)``
        are considered.  If too few points remain, the top half of the
        restricted range is used as a fallback.

    Returns
    -------
    bandgap_df : pandas.DataFrame
        Table summarising the extracted band gaps for each sample.  Contains
        columns ``Sample`` and ``Eg (eV)``.  Samples that could not be
        processed have ``NaN`` band gaps.
    tauc_curves : dict
        Mapping from sample name to a DataFrame with columns ``E (eV)`` and
        ``Tauc`` representing the smoothed Tauc curve used for plotting.
    fit_params : dict
        Mapping from sample name to a tuple ``(slope, intercept)`` of the
        fitted line.  Values are ``(np.nan, np.nan)`` when a fit is not
        possible or the slope is non‑positive.

    Notes
    -----
    - The absorption coefficient is calculated as ``alpha = -ln(T)/d``
      where ``T`` is the corrected transmittance (unitless) and ``d`` is
      the film thickness in metres.  This relation follows from Beer–Lambert's
      law and assumes negligible scattering and reflectance losses.
    - A glass correction divides the sample transmittance by the glass
      transmittance prior to computing ``alpha``.  Any zero or negative
      transmittance values are replaced with NaN to avoid invalid logs.
    """
    # Prepare result containers
    bandgap_records: List[Dict[str, Any]] = []
    tauc_curves: Dict[str, pd.DataFrame] = {}
    fit_params: Dict[str, Tuple[float, float]] = {}
    # Detailed data for each sample containing intermediate columns (wavelength, transmittance/absorbance, alpha, energy, Tauc)
    detailed_data: Dict[str, pd.DataFrame] = {}
    # Validate thickness dictionary; if None, skip all samples
    if thicknesses is None:
        for s in samples:
            bandgap_records.append({'Sample': s, 'Eg (eV)': np.nan})
            fit_params[s] = (np.nan, np.nan)
        return pd.DataFrame(bandgap_records), tauc_curves, fit_params, detailed_data
    # Extract wavelength column
    wl_col = ('global', 'Wavelength (nm)')
    if wl_col not in merged_data.columns:
        for s in samples:
            bandgap_records.append({'Sample': s, 'Eg (eV)': np.nan})
            fit_params[s] = (np.nan, np.nan)
        return pd.DataFrame(bandgap_records), tauc_curves, fit_params, detailed_data
    wavelengths = merged_data[wl_col].iloc[:limit_rows].astype(float)
    energies = 1240.0 / wavelengths
    # Get glass transmittance if provided
    glass_T = None
    if glass_sample is not None:
        glass_col = (glass_sample, '%T')
        if glass_col in merged_data.columns:
            glass_T = merged_data[glass_col].iloc[:limit_rows].astype(float) / 100.0
            # Replace non‑positive values with NaN to avoid division issues
            glass_T = glass_T.replace(0, np.nan)
            glass_T = glass_T.where(glass_T > 0, np.nan)
        else:
            glass_T = None
    # Ensure smoothing window is odd and greater than polyorder
    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window <= smooth_polyorder:
        smooth_window = smooth_polyorder + 3
        if smooth_window % 2 == 0:
            smooth_window += 1
    # Process each sample
    for sample in samples:
        # Skip the glass sample itself if present in sample list
        if sample == glass_sample:
            continue
        # Attempt to locate a transmittance or absorbance column for this sample
        # Candidate keys for transmittance (case insensitive)
        # Candidate keys for transmittance and absorbance measurements.  These lists
        # include various possible column headers users may use (case sensitive).
        trans_keys = [
            '%T', 'T%', 'T', 't%', 't',
            'Transmittance', 'transmittance',
            'Transmittance (%)', 'Transmittance %',
            'Transmission', 'Transmission (%)', 'Transmission %', 'transmission', 'transmission (%)',
            'Trans', 'trans',
        ]
        abs_keys = [
            '%A', 'A%', 'Abs', 'Absorption', 'Absorbance', 'A', 'a',
            'Absorption (%)', 'Absorbance (%)', 'Absorption %', 'Absorbance %',
        ]
        t_col = None
        a_col = None
        # Search for transmittance column
        for key in trans_keys:
            candidate = (sample, key)
            if candidate in merged_data.columns:
                t_col = candidate
                break
        # Search for absorbance/absorption column if no transmittance found
        if t_col is None:
            for key in abs_keys:
                candidate = (sample, key)
                if candidate in merged_data.columns:
                    a_col = candidate
                    break
        # If neither transmittance nor absorbance data available, skip sample
        if t_col is None and a_col is None:
            bandgap_records.append({'Sample': sample, 'Eg (eV)': np.nan})
            fit_params[sample] = (np.nan, np.nan)
            continue
        # Compute absorption coefficient alpha depending on available data and per‑sample thickness
        # Validate thickness for this sample
        thickness_nm = thicknesses.get(sample)
        if thickness_nm is None or thickness_nm <= 0:
            bandgap_records.append({'Sample': sample, 'Eg (eV)': np.nan})
            fit_params[sample] = (np.nan, np.nan)
            continue
        # Convert thickness to metres for this sample
        d_m = thickness_nm * 1e-9
        alpha = None
        # Prepare variables for transmittance and absorbance to build detailed table
        trans_fraction = None  # Transmittance as a fraction (0–1)
        raw_trans_percent = None  # Transmittance in percent (%), if available
        absorbance_vals = None  # Absorbance (base‑10), either measured or derived
        if t_col is not None:
            # Use transmittance.  Convert to fraction and apply glass correction.
            raw_trans_percent = merged_data[t_col].iloc[:limit_rows].astype(float)
            T_sample = raw_trans_percent / 100.0
            # Replace non‑positive values with NaN
            T_sample = T_sample.where(T_sample > 0, np.nan)
            # Correct with glass transmittance if available
            if glass_T is not None:
                # Align lengths
                if len(glass_T) < len(T_sample):
                    glass_vals = glass_T
                else:
                    glass_vals = glass_T.iloc[:limit_rows]
                with np.errstate(divide='ignore', invalid='ignore'):
                    trans_fraction = T_sample / glass_vals
            else:
                trans_fraction = T_sample
            # Compute absorbance from transmittance (base‑10)
            with np.errstate(divide='ignore', invalid='ignore'):
                absorbance_vals = -np.log10(trans_fraction)
            # Compute alpha using Beer–Lambert law: alpha = −ln(T)/d
            with np.errstate(divide='ignore', invalid='ignore'):
                alpha = -np.log(trans_fraction) / d_m
        elif a_col is not None:
            # Use absorbance/absorption data.  Assume base‑10 units: alpha = 2.303 * A / d
            absorbance_vals = merged_data[a_col].iloc[:limit_rows].astype(float)
            # Replace non‑finite values with NaN
            absorbance_vals = absorbance_vals.replace([np.inf, -np.inf], np.nan)
            # Convert to transmittance fraction: T = 10^(−A)
            with np.errstate(divide='ignore', invalid='ignore'):
                trans_fraction = 10 ** (-absorbance_vals)
            raw_trans_percent = trans_fraction * 100.0
            # Compute alpha directly; values remain NaN where absorbance_vals is NaN
            with np.errstate(divide='ignore', invalid='ignore'):
                alpha = (2.303 * absorbance_vals) / d_m
        # If alpha could not be computed, skip sample
        if alpha is None or np.all(np.isnan(alpha)):
            bandgap_records.append({'Sample': sample, 'Eg (eV)': np.nan})
            fit_params[sample] = (np.nan, np.nan)
            continue
        # Compute Tauc y‑values (unsmoothed): (alpha * E)^n
        with np.errstate(invalid='ignore'):
            base_vals = alpha * energies
            y_vals_raw = np.power(base_vals, exponent)
        # Build detailed data table for this sample.  Always include
        # wavelength, transmittance in % and fraction, absorbance, alpha, energy and
        # (alpha*E)^n.  Some values may be NaN if not applicable.
        try:
            detailed_df = pd.DataFrame({
                'Wavelength (nm)': wavelengths,
                'Transmittance (%)': raw_trans_percent,
                'Transmittance (fraction)': trans_fraction,
                'Absorbance (base-10)': absorbance_vals,
                'Absorption coefficient (1/m)': alpha,
                'Energy (eV)': energies,
                '(αE)^n': y_vals_raw,
            })
        except Exception:
            # Fallback: minimal columns
            detailed_df = pd.DataFrame({
                'Wavelength (nm)': wavelengths,
                'Absorption coefficient (1/m)': alpha,
                'Energy (eV)': energies,
                '(αE)^n': y_vals_raw,
            })
        # Drop rows with invalid values in Tauc_raw
        tauc_df = pd.DataFrame({'E (eV)': energies, 'Tauc_raw': y_vals_raw})
        # Drop rows with invalid values
        tauc_df = tauc_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        if tauc_df.empty:
            bandgap_records.append({'Sample': sample, 'Eg (eV)': np.nan})
            fit_params[sample] = (np.nan, np.nan)
            detailed_data[sample] = detailed_df
            continue
        # Apply Savitzky–Golay smoothing
        try:
            smoothed = savgol_filter(
                tauc_df['Tauc_raw'].values,
                window_length=smooth_window,
                polyorder=smooth_polyorder,
            )
        except Exception:
            smoothed = tauc_df['Tauc_raw'].values
        tauc_df['Tauc_smooth'] = smoothed
        # Determine energy range for fitting.  If a specific range is provided, use it.
        # Otherwise, use the full range of energies for this sample.
        if energy_range is not None:
            e_min, e_max = energy_range
        else:
            e_min = float(np.nanmin(energies))
            e_max = float(np.nanmax(energies))
        mask_range = (tauc_df['E (eV)'] >= e_min) & (tauc_df['E (eV)'] <= e_max)
        df_range = tauc_df.loc[mask_range].copy().reset_index(drop=True)
        if df_range.empty:
            bandgap_records.append({'Sample': sample, 'Eg (eV)': np.nan})
            fit_params[sample] = (np.nan, np.nan)
            continue
        # Determine threshold based on smoothed Tauc values
        max_val = df_range['Tauc_smooth'].max()
        if not np.isfinite(max_val) or max_val <= 0:
            bandgap_records.append({'Sample': sample, 'Eg (eV)': np.nan})
            fit_params[sample] = (np.nan, np.nan)
            continue
        threshold = threshold_fraction * max_val
        mask_threshold = df_range['Tauc_smooth'] >= threshold
        df_fit = df_range.loc[mask_threshold]
        # Fallback: if too few points, take top half by Tauc
        if df_fit.shape[0] < 2:
            df_fit = df_range.sort_values('Tauc_smooth', ascending=False).iloc[: max(2, df_range.shape[0] // 2)]
        # Fit a straight line: y = m x + b
        try:
            x_vals = df_fit['E (eV)'].values
            y_vals_fit = df_fit['Tauc_smooth'].values
            coeffs = np.polyfit(x_vals, y_vals_fit, 1)
            slope, intercept = coeffs[0], coeffs[1]
            if slope > 0:
                Eg = -intercept / slope
            else:
                Eg = np.nan
        except Exception:
            slope, intercept = np.nan, np.nan
            Eg = np.nan
        bandgap_records.append({'Sample': sample, 'Eg (eV)': Eg})
        fit_params[sample] = (slope, intercept)
        # Prepare smoothed Tauc curve for plotting
        tauc_plot_df = df_range[['E (eV)', 'Tauc_smooth']].rename(columns={'Tauc_smooth': 'Tauc'})
        tauc_curves[sample] = tauc_plot_df
        # Store detailed data for this sample (unsmoothed and intermediate values)
        detailed_data[sample] = detailed_df
    bandgap_df = pd.DataFrame(bandgap_records)
    return bandgap_df, tauc_curves, fit_params, detailed_data


def fit_tauc_line(
    energies: np.ndarray,
    tauc_values: np.ndarray,
    method: str = 'auto',
    threshold_fraction: float = 0.1,
    energy_window: Optional[Tuple[float, float]] = None,
    min_points: int = 2,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Fit a straight line to the Tauc curve and return fit parameters and diagnostics.

    Returns (slope, intercept, Eg, R2, used_mask)
    """
    # Ensure numpy arrays
    E = np.array(energies, dtype=float)
    Y = np.array(tauc_values, dtype=float)
    used = np.isfinite(E) & np.isfinite(Y)
    if energy_window is not None:
        lo, hi = energy_window
        used &= (E >= lo) & (E <= hi)
    if not np.any(used):
        return np.nan, np.nan, np.nan, np.nan, used
    E_used = E[used]
    Y_used = Y[used]
    # Automatic selection by threshold
    if method == 'auto':
        try:
            max_val = np.nanmax(Y_used)
            threshold = threshold_fraction * max_val
            mask_thr = Y_used >= threshold
            if np.sum(mask_thr) < min_points:
                # fallback: top half of points by Y magnitude
                order = np.argsort(Y_used)[::-1]
                take = max(min_points, len(Y_used) // 2)
                idx = order[:take]
                sel = np.zeros_like(Y_used, dtype=bool)
                sel[idx] = True
            else:
                sel = mask_thr
        except Exception:
            sel = np.ones_like(Y_used, dtype=bool)
    else:
        # manual: use all points in E_used (caller applies window)
        sel = np.ones_like(Y_used, dtype=bool)
    if np.sum(sel) < min_points:
        return np.nan, np.nan, np.nan, np.nan, used
    x = E_used[sel]
    y = Y_used[sel]
    try:
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
        Eg = -intercept / slope if slope != 0 and slope > 0 else np.nan
    except Exception:
        slope, intercept, Eg, r2 = np.nan, np.nan, np.nan, np.nan
    # Build final used mask in the original E array
    used_mask = np.zeros_like(used, dtype=bool)
    # Map sel indices back to used positions
    used_indices = np.where(used)[0]
    selected_indices = used_indices[np.where(sel)[0]]
    used_mask[selected_indices] = True
    return slope, intercept, Eg, r2, used_mask


def plot_tauc_curve(
    tauc_df: pd.DataFrame,
    sample: str,
    fit_params: Tuple[float, float],
    Eg: float,
    color: Any,
    line_style: str,
    line_width: float,
    marker: str,
    legend_loc: str,
    show_grid: bool,
    figsize: Tuple[float, float],
    dpi: int,
) -> Any:
    """
    Create a Tauc plot figure for a single sample with optional linear fit
    line and band gap annotation.

    Args:
        tauc_df: DataFrame containing the Tauc curve with columns ``E (eV)``
            and ``Tauc``. Assumed to be sorted by energy.
        sample: Name of the sample for labelling.
        fit_params: Tuple ``(slope, intercept)`` of the fitted line. If either
            value is ``np.nan`` no fit line will be plotted.
        Eg: Extracted band gap energy for annotation. If ``np.nan`` the band
            gap annotation will be omitted.
        color: Colour to use for the Tauc curve. Accepts any matplotlib
            colour (e.g., RGB tuple, hex string).
        line_style: Line style for the Tauc curve.
        line_width: Line width for the Tauc curve.
        marker: Marker style for the Tauc curve. Use ``'None'`` for no
            markers.
        legend_loc: Location for legend placement.
        show_grid: Whether to display grid lines.
        figsize: Tuple specifying figure width and height in inches.
        dpi: Resolution of the figure in dots per inch.

    Returns:
        A matplotlib Figure object containing the Tauc plot.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Plot the Tauc curve
    if marker == 'None':
        mk = None
    else:
        mk = marker
    ax.plot(
        tauc_df['E (eV)'],
        tauc_df['Tauc'],
        linestyle=line_style,
        linewidth=line_width,
        marker=mk,
        color=color,
        label=f'{sample} Tauc',
    )
    # Plot fitted line if available
    slope, intercept = fit_params
    if np.isfinite(slope) and np.isfinite(intercept) and slope > 0:
        x_fit = tauc_df['E (eV)']
        y_fit = slope * x_fit + intercept
        ax.plot(
            x_fit,
            y_fit,
            color='r',
            linestyle='--',
            linewidth=1.5,
            label=f'Linear fit',
        )
    # Annotate band gap if finite
    if np.isfinite(Eg) and Eg > 0:
        ax.axvline(Eg, color='k', linestyle=':', linewidth=1.5)
        ax.text(
            Eg,
            0.95 * np.nanmax(tauc_df['Tauc']),
            f'E$_g$ = {Eg:.3f} eV',
            color='k',
            ha='center',
            va='top',
            bbox=dict(facecolor='white', alpha=0.7, lw=0),
        )
    # Labels and style
    ax.set_xlabel('Photon energy E (eV)')
    ax.set_ylabel(r'$(F(R) \times E)^{n}$')
    ax.set_title(f'Tauc plot ({sample})')
    if show_grid:
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc=legend_loc)
    return fig


def main() -> None:
    """Entry point for the Streamlit UV‑Vis analysis app."""
    st.set_page_config(page_title='UV‑Vis Analysis', layout='wide')
    # Verify that plotting backends are available. If not, show a clear
    # message and provide guidance to add missing packages to
    # `requirements.txt` in the repository (Streamlit Cloud will install
    # packages listed there on deploy).
    missing_pkgs: List[str] = []
    if plt is None:
        missing_pkgs.append('matplotlib')
    # If scipy's functions are the fallback wrappers defined above they
    # will raise RuntimeError at use; detect absence by checking the
    # module presence instead.
    try:
        import scipy  # type: ignore
    except Exception:
        missing_pkgs.append('scipy')
    if missing_pkgs:
        st.title('UV‑Vis Spectroscopy Analysis')
        st.error(
            'The application is missing required Python packages: '
            + ', '.join(missing_pkgs)
            + '.\nPlease add them to `requirements.txt` and redeploy/restart the app.'
        )
        st.markdown(
            """
            Suggested `requirements.txt` entries:
            ```
            matplotlib
            scipy
            streamlit
            pandas
            numpy
            openpyxl
            ```
            """
        )
        return
    st.title('UV‑Vis Spectroscopy Analysis')

    st.markdown(
        """
        Upload a CSV file containing UV‑Vis spectroscopy measurements to analyse
        transmission, reflectance and absorption curves. The CSV should
        contain a multi‑index header with the top level naming each sample and
        the second level specifying the measurement (``%T`` for
        transmission and ``%R`` for reflectance). The first column should
        correspond to wavelength values.
        """
    )

    # File upload: allow multiple files of various supported types
    uploaded_files = st.file_uploader(
        'Upload one or more data files (CSV, Excel or TXT)',
        type=['csv', 'txt', 'xls', 'xlsx'],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        st.info(
            'Please upload at least one data file to begin. You can drag and drop '
            'or use the browser above. Supported formats include CSV, TXT, XLS and XLSX.'
        )
        return

    # Parse each uploaded file into a MultiIndex DataFrame
    parsed_dfs: List[pd.DataFrame] = []
    parse_errors: List[str] = []
    for f in uploaded_files:
        df_parsed = read_uvvis_file(f)
        if df_parsed is not None:
            parsed_dfs.append(df_parsed)
        else:
            parse_errors.append(f.name)
    if not parsed_dfs:
        st.error('None of the uploaded files could be parsed. Please check their format.')
        return
    if parse_errors:
        st.warning(
            f'The following files could not be parsed and were skipped: {", ".join(parse_errors)}'
        )

    # Merge the parsed DataFrames on the wavelength
    data = merge_dataframes(parsed_dfs)
    if data is None:
        st.error('Failed to merge the uploaded data. Ensure the files contain wavelength information.')
        return
    # Determine available samples from the merged DataFrame (excluding the global wavelength)
    sample_candidates = sorted({col[0] for col in data.columns if col[0] != 'global'})
    if not sample_candidates:
        st.error('No samples detected in the uploaded data. Ensure the files include sample information.')
        return
    # Show a preview of the raw merged data
    with st.expander('Preview merged data'):
        st.dataframe(data.head())
    # Display parsed data by sample in a collapsible interface. Users can
    # expand individual samples to inspect their wavelength and measurement
    # columns before deciding whether to include them in analyses. A
    # checkbox per sample allows the user to select or deselect that
    # sample from further processing.  Parsed data is shown in full so
    # users can review the entire available dataset rather than just a
    # truncated preview.
    selected_samples: List[str] = []
    with st.expander('Parsed data by sample (expand to view)', expanded=False):
        for sample in sample_candidates:
            # Determine relevant columns for this sample
            sub_cols: List[Tuple[str, str]] = []
            # Always include the wavelength column
            if ('global', 'Wavelength (nm)') in data.columns:
                sub_cols.append(('global', 'Wavelength (nm)'))
            # Include %T, %R, %A columns if available
            for meas in ['%T', '%R', '%A']:
                col = (sample, meas)
                if col in data.columns:
                    sub_cols.append(col)
            if not sub_cols:
                continue
            with st.expander(f'{sample} data', expanded=False):
                # Checkbox to include or exclude this sample
                include_sample = st.checkbox(
                    f'Include {sample}', value=True, key=f'include_{sample}'
                )
                # Display the full subset of data for this sample
                df_sub = data.loc[:, sub_cols]
                st.dataframe(df_sub)
                if include_sample:
                    selected_samples.append(sample)

    # Sidebar for analysis options and customisation
    with st.sidebar:
        st.header('Analysis Settings')
        # Sample selection is handled via checkboxes in the parsed data section,
        # so we no longer show a multiselect here.  The variable
        # ``selected_samples`` is populated earlier from the parsed data
        # checkboxes.
        st.markdown('---')
        # Smoothing parameters
        st.subheader('Smoothing')
        window_length = st.number_input(
            'Savitzky–Golay window length (odd)',
            min_value=5,
            max_value=301,
            value=55,
            step=2,
            help='Must be odd and greater than the polynomial order.',
        )
        polyorder = st.number_input(
            'Polynomial order',
            min_value=1,
            max_value=5,
            value=2,
            step=1,
        )
        st.markdown('---')
        # Derivative peak detection range
        st.subheader('Peak Detection Range')
        col1, col2 = st.columns(2)
        xmin = col1.number_input('Minimum wavelength (nm)', value=600.0, step=1.0)
        xmax = col2.number_input('Maximum wavelength (nm)', value=900.0, step=1.0)
        if xmin >= xmax:
            st.warning('The minimum wavelength must be less than the maximum wavelength.')
        st.markdown('---')
        # Limit rows processed
        # Default to processing up to 883 rows as in the original script (or less
        # if the dataset is shorter). This preserves backwards compatibility
        # with the original pipeline while allowing the user to adjust it.
        default_rows = min(len(data), 883)
        limit_rows = st.number_input(
            'Number of rows to process',
            min_value=1,
            max_value=len(data),
            value=default_rows,
            step=1,
            help=(
                'Limiting the number of rows can improve performance for very large files. '
                'The default uses 883 rows to match the original analysis script.'
            ),
        )
        st.markdown('---')
        # Plot selection
        st.subheader('Plots to display')
        show_transmission = st.checkbox('Transmission vs Wavelength', value=True)
        show_reflectance = st.checkbox('Reflectance vs Wavelength', value=True)
        show_absorption = st.checkbox('Absorption vs Wavelength', value=True)
        show_normalised = st.checkbox('Normalised Absorption', value=True)
        show_first_derivative = st.checkbox('First Derivative', value=False)
        show_second_derivative = st.checkbox('Second Derivative with Peaks', value=False)
        # Tauc plot option
        show_tauc = st.checkbox(
            'Band gap analysis (Tauc plot)',
            value=False,
            help='Compute band gaps using a Tauc plot for the selected samples.',
        )
        # Diagnostics and fallback behaviour
        enable_fallback_fit = st.checkbox(
            'Enable fallback fitting (try multiple thresholds)',
            value=True,
            help='If the initial fit fails or produces non-positive slope, try looser thresholds as fallback.',
        )
        show_diagnostics = st.checkbox(
            'Show diagnostic info for Tauc fits',
            value=False,
            help='When enabled, per-sample fit diagnostics (slope/intercept/used threshold) are shown in the per-sample expanders.',
        )
        st.markdown('---')
        # Plot customisation options
        st.subheader('Plot Customisation')
        # List of colormap names; choose a few perceptually diverse options
        colormap_options = [
            'copper', 'viridis', 'plasma', 'magma', 'cividis', 'turbo',
            'tab10', 'Set2', 'Spectral', 'rainbow'
        ]
        cmap_name = st.selectbox('Colour map', options=colormap_options, index=0)
        line_style_options = {
            'Solid': '-',
            'Dashed': '--',
            'Dash‑dot': '-.',
            'Dotted': ':'
        }
        line_style_key = st.selectbox('Line style', options=list(line_style_options.keys()), index=0)
        line_style = line_style_options[line_style_key]
        line_width = st.slider('Line width', min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        marker_options = ['None', 'o', 's', '^', 'D']
        marker = st.selectbox('Marker style', options=marker_options, index=0)
        # Figure size and DPI
        fig_width = st.number_input(
            'Figure width (inches)',
            min_value=4.0,
            max_value=20.0,
            value=10.0,
            step=0.5,
        )
        fig_height = st.number_input(
            'Figure height (inches)',
            min_value=3.0,
            max_value=15.0,
            value=6.0,
            step=0.5,
        )
        dpi = st.number_input(
            'Figure resolution (DPI)',
            min_value=72,
            max_value=600,
            value=300,
            step=10,
        )
        show_grid = st.checkbox('Show grid lines', value=True)
        legend_locations = [
            'best', 'upper right', 'upper left', 'lower right', 'lower left',
            'center left', 'center right', 'lower center', 'upper center', 'center'
        ]
        legend_loc = st.selectbox('Legend location', options=legend_locations, index=0)

    # Validate selection
    if not selected_samples:
        st.warning('Please select at least one sample to analyse.')
        return

    # Automatically process data when valid selections are made
    if xmin < xmax:
        with st.spinner('Processing data…'):
            merged_data, peak_values = process_data(
                data,
                samples=selected_samples,
                limit_rows=int(limit_rows),
                window_length=int(window_length),
                polyorder=int(polyorder),
                derivative_range=(xmin, xmax),
            )
        # Display peak values table and summary
        st.subheader('Detected peak values')
        st.dataframe(peak_values)
        for _, row in peak_values.iterrows():
            sample = row['Sample']
            wl = row['Wavelength (nm)']
            energy = row['Energy (eV)']
            if pd.notna(wl) and pd.notna(energy):
                st.markdown(
                    f"**{sample}**: largest peak at **{wl:.2f} nm** → **{energy:.2f} eV**"
                )
        # Download peak values as CSV
        csv_buffer = io.StringIO()
        peak_values.to_csv(csv_buffer, index=False)
        st.download_button(
            label='Download peak values as CSV',
            data=csv_buffer.getvalue(),
            file_name='peak_values.csv',
            mime='text/csv',
        )
        st.markdown('---')
        # Generate and display requested plots with customisation
        # Each plot is accompanied by a download button for high‑resolution output
        def display_and_download(fig: Any, plot_name: str):
            """Helper to display a figure and provide a download button."""
            st.pyplot(fig)
            buf = io.BytesIO()
            # Save using the specified DPI for high quality
            fig.savefig(buf, format='png', dpi=int(dpi))
            buf.seek(0)
            st.download_button(
                label=f'Download {plot_name} plot (PNG)',
                data=buf.getvalue(),
                file_name=f'{plot_name.lower().replace(" ", "_")}.png',
                mime='image/png',
            )

        if show_transmission:
            fig = plot_lines(
                merged_data,
                samples=selected_samples,
                column_suffix='%T',
                y_label='% Transmission',
                title='Transmission vs Wavelength',
                cmap_name=cmap_name,
                line_style=line_style,
                line_width=line_width,
                marker=marker,
                legend_loc=legend_loc,
                show_grid=show_grid,
                figsize=(fig_width, fig_height),
                dpi=int(dpi),
            )
            display_and_download(fig, 'Transmission')
        if show_reflectance:
            fig = plot_lines(
                merged_data,
                samples=selected_samples,
                column_suffix='%R',
                y_label='% Reflectance',
                title='Reflectance vs Wavelength',
                cmap_name=cmap_name,
                line_style=line_style,
                line_width=line_width,
                marker=marker,
                legend_loc=legend_loc,
                show_grid=show_grid,
                figsize=(fig_width, fig_height),
                dpi=int(dpi),
            )
            display_and_download(fig, 'Reflectance')
        if show_absorption:
            fig = plot_lines(
                merged_data,
                samples=selected_samples,
                column_suffix='%A',
                y_label='% Absorption',
                title='Absorption vs Wavelength',
                cmap_name=cmap_name,
                line_style=line_style,
                line_width=line_width,
                marker=marker,
                legend_loc=legend_loc,
                show_grid=show_grid,
                figsize=(fig_width, fig_height),
                dpi=int(dpi),
            )
            display_and_download(fig, 'Absorption')
        if show_normalised:
            fig = plot_lines(
                merged_data,
                samples=selected_samples,
                column_suffix='%A Normalized',
                y_label='Normalised Absorption',
                title='Normalised Absorption vs Wavelength',
                xlim=(xmin, xmax),
                cmap_name=cmap_name,
                line_style=line_style,
                line_width=line_width,
                marker=marker,
                legend_loc=legend_loc,
                show_grid=show_grid,
                figsize=(fig_width, fig_height),
                dpi=int(dpi),
            )
            display_and_download(fig, 'Normalised_Absorption')
        if show_first_derivative:
            fig = plot_lines(
                merged_data,
                samples=selected_samples,
                column_suffix='%A Normalized 1st Derivative',
                y_label='1st Derivative (normalised)',
                title='First Derivative of Normalised Absorption',
                xlim=(xmin, xmax),
                cmap_name=cmap_name,
                line_style=line_style,
                line_width=line_width,
                marker=marker,
                legend_loc=legend_loc,
                show_grid=show_grid,
                figsize=(fig_width, fig_height),
                dpi=int(dpi),
            )
            display_and_download(fig, 'First_Derivative')
        if show_second_derivative:
            fig = plot_lines(
                merged_data,
                samples=selected_samples,
                column_suffix='%A Normalized 2nd Derivative',
                y_label='2nd Derivative (normalised)',
                title='Second Derivative of Normalised Absorption',
                xlim=(xmin, xmax),
                annotate_peaks=True,
                peak_values=peak_values,
                cmap_name=cmap_name,
                line_style=line_style,
                line_width=line_width,
                marker=marker,
                legend_loc=legend_loc,
                show_grid=show_grid,
                figsize=(fig_width, fig_height),
                dpi=int(dpi),
            )
            display_and_download(fig, 'Second_Derivative')

        

        # ------------------------------------------------------------------
        # Band gap data preview (tables, no plotting)
        # For each selected sample show a table with wavelength, the chosen
        # measurement column (e.g. %T or %R), computed alpha (or F(R)) and
        # photon energy E (eV). User can choose the measurement column and
        # optionally provide thickness (nm) used for alpha calculation.
        if show_tauc:
            st.subheader('Band gap data preview (no plotting)')
            st.markdown(
                """
                For each sample/variation below choose the measurement column to use.
                If transmittance is chosen you may optionally provide a thickness (nm)
                to compute an absorption coefficient alpha = -ln(T)/d (1/m).
                If thickness is left as 0 the table will show -ln(T) (unitless).
                """
            )
            # Let user choose direct or indirect bandgap exponent
            st.info('Choose bandgap type for computing (αhν)^n')
            bg_type = st.radio('Bandgap type', options=['Direct (n=2)', 'Indirect (n=0.5)'], index=0)
            tauc_exponent = 2.0 if 'Direct' in bg_type else 0.5
            selected_for_tauc: List[str] = []
            # Locate wavelength column robustly. merged_data may be the flat
            # DataFrame produced by process_data (string column 'Wavelength (nm)')
            # or a MultiIndex DataFrame with ('global','Wavelength (nm)').
            wavelengths = None
            if 'Wavelength (nm)' in merged_data.columns:
                try:
                    wavelengths = merged_data['Wavelength (nm)'].iloc[:int(limit_rows)].astype(float).reset_index(drop=True)
                except Exception:
                    wavelengths = merged_data['Wavelength (nm)'].iloc[:int(limit_rows)].reset_index(drop=True).astype(float)
            else:
                wl_candidates = [c for c in merged_data.columns if isinstance(c, tuple) and c[1] == 'Wavelength (nm)']
                if wl_candidates:
                    try:
                        wavelengths = merged_data[wl_candidates[0]].iloc[:int(limit_rows)].astype(float).reset_index(drop=True)
                    except Exception:
                        wavelengths = merged_data[wl_candidates[0]].iloc[:int(limit_rows)].reset_index(drop=True)
            # Final fallback: first column cast to float
            if wavelengths is None:
                wavelengths = pd.to_numeric(merged_data.iloc[:, 0].iloc[:int(limit_rows)].reset_index(drop=True), errors='coerce')

            for s in selected_samples:
                # Discover available measurement columns for this sample.
                # Support both MultiIndex columns (sample, meas) and flat columns like 'Sample %T'.
                if any(isinstance(c, tuple) for c in merged_data.columns):
                    meas_cols = [c for c in merged_data.columns if isinstance(c, tuple) and c[0] == s and c[1]]
                    meas_names = [c[1] for c in meas_cols]
                    is_multi = True
                else:
                    prefix = f'{s} '
                    meas_cols = [c for c in merged_data.columns if isinstance(c, str) and c.startswith(prefix)]
                    # Derive measurement names by stripping the sample prefix
                    meas_names = [c[len(prefix):] for c in meas_cols]
                    is_multi = False
                if not meas_names:
                    continue
                with st.expander(f'{s} — data / choose measurement', expanded=False):
                    chosen = st.selectbox(f'Choose measurement column for {s}', options=meas_names, index=0, key=f'meas_{s}')
                    thickness_nm = st.number_input(f'Thickness for {s} (nm, 0 = unknown)', min_value=0.0, value=0.0, step=1.0, key=f'thick_{s}')
                    use_sample = st.checkbox(f'Select {s} for further analysis', value=False, key=f'use_{s}')
                    if use_sample:
                        selected_for_tauc.append(s)
                    # Retrieve measurement series robustly depending on column layout
                    try:
                        if is_multi:
                            chosen_col = (s, chosen)
                            meas_series = merged_data[chosen_col].iloc[:int(limit_rows)].astype(float).reset_index(drop=True)
                        else:
                            chosen_col_str = f'{s} {chosen}'
                            meas_series = merged_data[chosen_col_str].iloc[:int(limit_rows)].astype(float).reset_index(drop=True)
                    except Exception:
                        meas_series = pd.Series([np.nan] * len(wavelengths))
                    name_lower = str(chosen).lower()
                    is_trans = any(tok in name_lower for tok in ['%t', 't%', 'trans', 'transmittance', 'transmission'])
                    is_refl = any(tok in name_lower for tok in ['%r', 'r%', 'reflect', 'reflectance'])
                    if is_trans:
                        # Detect whether transmission values are in percent (0-100)
                        # or in fraction (0-1). Use a heuristic: if values > 1.5 -> percent.
                        meas_median = pd.Series(meas_series).abs().median(skipna=True)
                        if pd.notna(meas_median) and meas_median > 1.5:
                            meas_frac = meas_series / 100.0
                            meas_format = 'percent'
                        else:
                            meas_frac = meas_series.copy()
                            meas_format = 'fraction'
                        meas_frac = meas_frac.where(meas_frac > 0, np.nan)
                        if thickness_nm and thickness_nm > 0:
                            d_m = thickness_nm * 1e-9
                            with np.errstate(divide='ignore', invalid='ignore'):
                                alpha_vals = -np.log(meas_frac) / d_m
                            alpha_name = 'α (1/m) — α = -ln(T)/d'
                        else:
                            with np.errstate(divide='ignore', invalid='ignore'):
                                alpha_vals = -np.log(meas_frac)
                            alpha_name = 'α (unitless) — α = -ln(T)'
                    elif is_refl:
                        R = meas_series / 100.0
                        R_safe = R.where(R > 0, np.nan)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            alpha_vals = ((1.0 - R_safe) ** 2) / (2.0 * R_safe)
                        alpha_name = 'F(R)'
                    else:
                        alpha_vals = meas_series
                        alpha_name = 'Value'
                    with np.errstate(divide='ignore', invalid='ignore'):
                        energy = 1240.0 / wavelengths
                    # Compute alpha*hν and (alpha*hν)^n for all rows
                    # Ensure alpha_vals and energy are aligned as Series
                    alpha_s = pd.Series(alpha_vals).reset_index(drop=True)
                    energy_s = pd.Series(energy).reset_index(drop=True)
                    with np.errstate(invalid='ignore'):
                        alpha_hv = alpha_s * energy_s
                        alpha_hv_n = np.power(alpha_hv, tauc_exponent)

                    # Add a column indicating whether input T was percent or fraction when applicable
                    meta_cols = {}
                    if is_trans:
                        meta_cols['Transmittance format'] = meas_format

                    display_df = pd.DataFrame({
                        'Wavelength (nm)': wavelengths,
                        chosen: meas_series,
                        'Transmittance format': meta_cols.get('Transmittance format', ''),
                        alpha_name: alpha_s,
                        'α·hν': alpha_hv,
                        f'(α·hν)^{tauc_exponent}': alpha_hv_n,
                        'Energy (eV)': energy_s,
                    })
                    st.write('Preview table (first 200 rows):')
                    # Show the full computed table as requested (not limited)
                    st.dataframe(display_df)
                    csv_buf = io.StringIO()
                    display_df.to_csv(csv_buf, index=False)
                    st.download_button(label=f'Download computed table for {s} (CSV)', data=csv_buf.getvalue(), file_name=f'computed_{s}.csv', mime='text/csv', key=f'dl_{s}')
            st.session_state['selected_for_tauc'] = selected_for_tauc
            st.info('Per-sample computed tables shown above. Use the checkboxes to mark which samples to include in subsequent analyses.')

            # ------------------------------------------------------------------
            # Tauc plotting: overlay (α·hν)^n vs E for selected samples
            # (inserted here so wavelengths and tauc_exponent are defined)
            # ------------------------------------------------------------------
            st.sidebar.markdown('---')
            st.sidebar.subheader('Tauc plot customisation')
            tauc_auto_fit = st.sidebar.checkbox('Auto-fit axes', value=True, key='tauc_auto_fit')
            tauc_xmin = st.sidebar.number_input('Tauc X min (eV)', value=0.5, step=0.1, key='tauc_xmin')
            tauc_xmax = st.sidebar.number_input('Tauc X max (eV)', value=4.0, step=0.1, key='tauc_xmax')
            tauc_ymin = st.sidebar.number_input('Tauc Y min (a.u.)', value=0.0, step=0.1, key='tauc_ymin')
            tauc_ymax = st.sidebar.number_input('Tauc Y max (a.u.)', value=1.0, step=0.1, key='tauc_ymax')
            tauc_cmap = st.sidebar.selectbox('Tauc colour map', options=colormap_options, index=0, key='tauc_cmap')
            tauc_line_style = st.sidebar.selectbox('Tauc line style', options=list(line_style_options.keys()), index=0, key='tauc_line_style')
            tauc_marker = st.sidebar.selectbox('Tauc marker', options=marker_options, index=0, key='tauc_marker')
            tauc_line_width = st.sidebar.slider('Tauc line width', min_value=0.5, max_value=4.0, value=1.2, step=0.1, key='tauc_line_width')

            # Build or retrieve per-sample computed tables in session_state
            if 'tauc_tables' not in st.session_state:
                st.session_state['tauc_tables'] = {}
            tauc_tables_local: Dict[str, pd.DataFrame] = {}
            for s in selected_samples:
                use_key = f'use_{s}'
                if use_key in st.session_state and not st.session_state[use_key]:
                    continue
                meas_key = f'meas_{s}'
                thick_key = f'thick_{s}'
                if meas_key not in st.session_state:
                    continue
                chosen = st.session_state[meas_key]
                thickness_nm = st.session_state.get(thick_key, 0.0)
                try:
                    if any(isinstance(c, tuple) for c in merged_data.columns):
                        meas_series = merged_data[(s, chosen)].iloc[:int(limit_rows)].astype(float).reset_index(drop=True)
                    else:
                        meas_series = merged_data[f'{s} {chosen}'].iloc[:int(limit_rows)].astype(float).reset_index(drop=True)
                except Exception:
                    meas_series = pd.Series([np.nan] * len(wavelengths))
                meas_median = pd.Series(meas_series).abs().median(skipna=True)
                if pd.notna(meas_median) and meas_median > 1.5:
                    meas_frac = meas_series / 100.0
                else:
                    meas_frac = meas_series.copy()
                meas_frac = meas_frac.where(meas_frac > 0, np.nan)
                if thickness_nm and thickness_nm > 0:
                    d_m = thickness_nm * 1e-9
                    with np.errstate(divide='ignore', invalid='ignore'):
                        alpha_vals = -np.log(meas_frac) / d_m
                else:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        alpha_vals = -np.log(meas_frac)
                energies_full = 1240.0 / pd.to_numeric(wavelengths)
                alpha_s = pd.Series(alpha_vals).reset_index(drop=True)
                energy_s = pd.Series(energies_full).reset_index(drop=True)
                with np.errstate(invalid='ignore'):
                    alpha_hv = alpha_s * energy_s
                    alpha_hv_n = np.power(alpha_hv, tauc_exponent)
                df_t = pd.DataFrame({
                    'Wavelength (nm)': wavelengths,
                    'Energy (eV)': energy_s,
                    'Alpha': alpha_s,
                    'Alpha_hv': alpha_hv,
                    f'Alpha_hv_n': alpha_hv_n,
                })
                tauc_tables_local[s] = df_t
                st.session_state['tauc_tables'][s] = df_t

            if tauc_tables_local:
                plot_samples = sorted(list(tauc_tables_local.keys()))
                chosen_plot_samples = st.multiselect('Samples to plot (overlay)', options=plot_samples, default=plot_samples, key='tauc_plot_samples')
                if chosen_plot_samples:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=int(dpi))
                    cmap = plt.get_cmap(tauc_cmap)
                    colours = cmap(np.linspace(0, 1, max(len(chosen_plot_samples), 1)))
                    # Prepare fit storage
                    if 'tauc_fits' not in st.session_state:
                        st.session_state['tauc_fits'] = {}
                    fit_method = st.sidebar.selectbox('Fit method', options=['Automatic', 'Manual'], index=0, key='tauc_fit_method')
                    threshold_fraction = st.sidebar.slider('Threshold fraction (auto)', min_value=0.01, max_value=0.5, value=0.1, step=0.01, key='tauc_threshold')
                    manual_e_min = st.sidebar.number_input('Manual fit E min (eV)', value=float(0.5), step=0.01, key='tauc_manual_min')
                    manual_e_max = st.sidebar.number_input('Manual fit E max (eV)', value=float(3.5), step=0.01, key='tauc_manual_max')
                    show_fit_lines = st.sidebar.checkbox('Show fit lines', value=True, key='tauc_show_fit')
                    show_eg = st.sidebar.checkbox('Show E_g annotations', value=True, key='tauc_show_eg')
                    for idx, s in enumerate(chosen_plot_samples):
                        df_t = tauc_tables_local[s]
                        plot_df = df_t.dropna(subset=['Energy (eV)', 'Alpha_hv_n']).reset_index(drop=True)
                        if plot_df.empty:
                            continue
                        ax.plot(
                            plot_df['Energy (eV)'],
                            plot_df['Alpha_hv_n'],
                            label=s,
                            color=colours[idx % len(colours)],
                            linestyle=line_style,
                            linewidth=float(tauc_line_width),
                            marker=(None if (tauc_marker == 'None') else tauc_marker),
                        )
                        # Fit
                        energies_arr = plot_df['Energy (eV)'].values
                        y_arr = plot_df['Alpha_hv_n'].values
                        if fit_method == 'Automatic':
                            slope, intercept, Eg, r2, used_mask = fit_tauc_line(energies_arr, y_arr, method='auto', threshold_fraction=float(threshold_fraction))
                        else:
                            slope, intercept, Eg, r2, used_mask = fit_tauc_line(energies_arr, y_arr, method='manual', energy_window=(manual_e_min, manual_e_max))
                        st.session_state['tauc_fits'][s] = {'slope': slope, 'intercept': intercept, 'Eg': Eg, 'R2': r2, 'used_mask': used_mask}
                        # Draw fit line and Eg marker
                        if show_fit_lines and np.isfinite(slope) and np.isfinite(intercept):
                            # Evaluate fit line across plotted energy range
                            xfit = np.linspace(np.nanmin(energies_arr), np.nanmax(energies_arr), 100)
                            yfit = slope * xfit + intercept
                            ax.plot(xfit, yfit, linestyle='--', color='red', linewidth=1.2)
                        if show_eg and np.isfinite(Eg):
                            ax.axvline(Eg, color='k', linestyle=':', linewidth=1.0)
                            ax.text(Eg, 0.95 * np.nanmax(plot_df['Alpha_hv_n']), f'Eg={Eg:.3f} eV', rotation=90, va='top', ha='center', bbox=dict(facecolor='white', alpha=0.7, lw=0))
                    ax.set_xlabel('Photon energy E (eV)')
                    ax.set_ylabel(f'(α·hν)^{{{tauc_exponent}}} (a.u.)')
                    ax.set_title('Tauc plot overlay')
                    ax.grid(True)
                    ax.legend()
                    if tauc_auto_fit:
                        try:
                            ax.relim()
                            ax.autoscale_view()
                        except Exception:
                            pass
                    else:
                        ax.set_xlim(float(tauc_xmin), float(tauc_xmax))
                        ax.set_ylim(float(tauc_ymin), float(tauc_ymax))
                    st.pyplot(fig)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=int(dpi))
                    buf.seek(0)
                    st.download_button('Download Tauc overlay (PNG)', data=buf.getvalue(), file_name='tauc_overlay.png', mime='image/png')


if __name__ == '__main__':
    # Launch the Streamlit application.  When executed via ``streamlit run``,
    # ``main()`` is called directly.  When the script is executed
    # standalone (e.g. double‑click), a new Streamlit server process is
    # spawned to avoid multiple runtime instances.
    import sys
    import os
    try:
        from streamlit import runtime  # type: ignore
    except ModuleNotFoundError:
        # Streamlit is not installed. Inform the user and run main() as a
        # fallback so at least the script does something sensible.
        print(
            'Streamlit is required to run this application. Please install it with '
            '"pip install streamlit" and then double‑click the script again.'
        )
        # Run the core logic in a plain Python context to provide some
        # functionality even without Streamlit.
        main()
    else:
        # If a runtime already exists then we are running under
        # ``streamlit run`` and should call main() directly.
        try:
            exists_fn = runtime.exists
        except Exception:
            # In older versions of Streamlit ``runtime.exists`` may not
            # exist. Fall back to inspecting the Streamlit context.
            def exists_fn() -> bool:
                try:
                    from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
                    return get_script_run_ctx() is not None
                except Exception:
                    return False
        if exists_fn():
            # Already running inside a Streamlit runtime: execute main
            main()
        else:
            # Not running under Streamlit; launch a new server as a
            # separate process. Use ``sys.executable -m streamlit run`` to
            # ensure the correct Python interpreter is used. ``subprocess``
            # is preferred over invoking ``stcli.main`` directly because
            # it spawns a new process and avoids conflicting runtimes.
            import subprocess
            script_path = os.path.abspath(__file__)
            try:
                # Spawn the Streamlit process detached from this one.
                subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', script_path])
            except Exception:
                # Fallback to using the ``streamlit`` command directly.
                subprocess.Popen(['streamlit', 'run', script_path])
            # Exit the current process so that only the new Streamlit
            # process remains. Without this exit the original process
            # would continue running and may interfere with the server.
            raise SystemExit