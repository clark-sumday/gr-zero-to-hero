"""
Colorblind-friendly color schemes and visualization patterns.
Uses high-contrast colors that work for all types of color blindness.
"""

# Colorblind-friendly palette (works for deuteranopia, protanopia, tritanopia)
# Based on Wong 2011 and IBM Design recommendations
COLORS = {
    'blue': '#0173B2',      # Strong blue
    'orange': '#DE8F05',    # Vivid orange
    'green': '#029E73',     # Bluish green
    'red': '#CC3311',       # Vermillion red
    'purple': '#9C3587',    # Purple
    'yellow': '#ECE133',    # Yellow
    'cyan': '#56B4E9',      # Sky blue
    'pink': '#CC79A7',      # Reddish purple
    'black': '#000000',
    'gray': '#888888'
}

# Line styles for additional distinction
LINE_STYLES = ['-', '--', '-.', ':']

# Marker styles
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

def get_color_cycle(n=None):
    """
    Get a list of colorblind-friendly colors.

    Args:
        n: Number of colors needed. If None, returns all colors.

    Returns:
        List of color hex codes
    """
    color_list = [
        COLORS['blue'],
        COLORS['orange'],
        COLORS['green'],
        COLORS['red'],
        COLORS['purple'],
        COLORS['cyan'],
        COLORS['yellow'],
        COLORS['pink']
    ]

    if n is None:
        return color_list
    return color_list[:n]

def setup_plot_style(ax=None):
    """
    Configure plot with colorblind-friendly defaults.

    Args:
        ax: Matplotlib axes object. If None, uses current axes.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    # Set color cycle
    ax.set_prop_cycle(color=get_color_cycle())

    # Use high contrast
    ax.set_facecolor('white')
    ax.spines['bottom'].set_color(COLORS['black'])
    ax.spines['left'].set_color(COLORS['black'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Grid with subtle gray
    ax.grid(True, alpha=0.3, color=COLORS['gray'], linewidth=0.5)

def get_distinct_style(index):
    """
    Get a distinct combination of color, linestyle, and marker.

    Args:
        index: Integer index for the style

    Returns:
        dict with 'color', 'linestyle', 'marker' keys
    """
    colors = get_color_cycle()
    return {
        'color': colors[index % len(colors)],
        'linestyle': LINE_STYLES[index % len(LINE_STYLES)],
        'marker': MARKERS[index % len(MARKERS)]
    }
