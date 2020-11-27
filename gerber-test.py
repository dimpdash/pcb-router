import gerber
from gerber.render import GerberCairoContext

# Read gerber and Excellon files
top_copper = gerber.read('copper.GTL')
nc_drill = gerber.read('ncdrill.txt')

# Rendering context
ctx = GerberCairoContext()

print(top_copper)

# Create SVG image
top_copper.render(ctx)
nc_drill.render(ctx, '/mnt/c/Users/dimpd/OneDrive/Documents/GitHub/pcb-router/composite.svg')