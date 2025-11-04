from PIL import Image, ImageDraw, ImageFont

def _hex_to_rgb(h, a=255): 
    """Converts hex to an (r, g, b, a) tuple."""
    return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (a,)

def get_font(size):
    """Tries to load a common sans-serif font, falling back to the default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        return ImageFont.load_default()

# --- PREVIEW DRAWING LIBRARY (18 UNIQUE DESIGNS) ---
def draw_sidebar(draw, t, w, h): draw.rectangle([15, 15, 30, h - 15], fill=t["accent_color"])
def draw_corner_accent(draw, t, w, h): draw.rectangle([w-120, 25, w-20, 35], fill=t["accent_color"]); draw.rectangle([w-35, 25, w-25, 100], fill=t["accent_color"])
def draw_header_lines(draw, t, w, h): draw.line([(50, 130), (w-50, 130)], fill=t["accent_color"], width=3)
def draw_gradient_bar(draw, t, w, h):
    start, end = _hex_to_rgb(t["accent_color"])[:3], _hex_to_rgb(t["font_color"])[:3]
    for i in range(w):
        r, g, b = [int(start[j] + (end[j]-start[j])*(i/w)) for j in range(3)]
        draw.line([(i, h-20), (i, h)], fill=(r,g,b), width=1)
def draw_border_frame(draw, t, w, h): draw.rectangle([0,0,w-1,h-1], outline=t["accent_color"], width=10)
def draw_title_underline(draw, t, w, h): draw.rectangle([50, 120, 350, 130], fill=t["accent_color"])
def draw_pillar_bars(draw, t, w, h):
    draw.rectangle([30,30,50,h-30], fill=_hex_to_rgb(t["accent_color"], 128))
    draw.rectangle([w-50,30,w-30,h-30], fill=_hex_to_rgb(t["accent_color"], 128))
def draw_diagonal_accent(draw, t, w, h): draw.polygon([(0,h-150), (250,h), (0,h)], fill=t["accent_color"])
def draw_bottom_left_block(draw, t, w, h): draw.rectangle([0,h-120, 200,h], fill=t["accent_color"])
def draw_top_right_triangle(draw, t, w, h): draw.polygon([(w-200,0), (w,0), (w,200)], fill=t["accent_color"])
def draw_chevron_bottom(draw, t, w, h): draw.polygon([(w/2-100,h-20), (w/2,h), (w/2+100,h-20), (w/2,h-40)], fill=t["accent_color"])
def draw_top_bottom_thin_bars(draw, t, w, h): draw.rectangle([0,10,w,15], fill=t["accent_color"]); draw.rectangle([0,h-15,w,h-10], fill=t["accent_color"])
def draw_left_edge_wave(draw, t, w, h): draw.ellipse([-60,0,80,h/2], fill=t["accent_color"]); draw.ellipse([-60,h/2,80,h], fill=t["accent_color"])
def draw_cross_lines(draw, t, w, h): draw.line([(50,h/2),(w-50,h/2)], fill=t["accent_color"], width=2); draw.line([(w/2,50),(w/2,h-50)], fill=_hex_to_rgb(t["accent_color"], 128), width=2)

# --- COMPOSITE (OVERLAY) DESIGNS ---
def draw_geometric_shapes(img, t, w, h):
    overlay = Image.new('RGBA', img.size); draw_o = ImageDraw.Draw(overlay)
    draw_o.ellipse([-100,-100,200,200], fill=_hex_to_rgb(t["accent_color"], 80)); draw_o.ellipse([w-150,h-100,w+100,h+150], fill=_hex_to_rgb(t["accent_color"], 80))
    return Image.alpha_composite(img.convert('RGBA'), overlay)
def draw_offset_blocks(img, t, w, h):
    overlay = Image.new('RGBA', img.size); draw_o = ImageDraw.Draw(overlay)
    draw_o.rectangle([w-150,h-150,w,h], fill=_hex_to_rgb(t["accent_color"], 100)); draw_o.rectangle([w-200,h-120,w-50,h+30], fill=_hex_to_rgb(t["accent_color"], 80))
    return Image.alpha_composite(img.convert('RGBA'), overlay)
def draw_faded_circle(img, t, w, h):
    overlay = Image.new('RGBA', img.size); draw_o = ImageDraw.Draw(overlay)
    draw_o.ellipse([w-400,-200,w+200,400], fill=_hex_to_rgb(t["accent_color"], 60))
    return Image.alpha_composite(img.convert('RGBA'), overlay)
def draw_dotted_background(img, t, w, h):
    overlay = Image.new('RGBA', img.size); draw_o = ImageDraw.Draw(overlay)
    for x in range(0,w,40):
        for y in range(0,h,40): draw_o.ellipse([x,y,x+5,y+5], fill=_hex_to_rgb(t["accent_color"], 100))
    return Image.alpha_composite(img.convert('RGBA'), overlay)

# --- PREVIEW DISPATCHER ---
PREVIEW_MAP = {
    "sidebar": draw_sidebar, "corner_accent": draw_corner_accent, "header_lines": draw_header_lines,
    "gradient_bar": draw_gradient_bar, "border_frame": draw_border_frame,
    "title_underline": draw_title_underline, "pillar_bars": draw_pillar_bars, "diagonal_accent": draw_diagonal_accent,
    "bottom_left_block": draw_bottom_left_block, "top_right_triangle": draw_top_right_triangle, 
    "chevron_bottom": draw_chevron_bottom, "top_bottom_thin_bars": draw_top_bottom_thin_bars,
    "left_edge_wave": draw_left_edge_wave, "cross_lines": draw_cross_lines
}
COMPOSITE_MAP = { "geometric_shapes": draw_geometric_shapes, "offset_blocks": draw_offset_blocks, "faded_circle": draw_faded_circle, "dotted_background": draw_dotted_background}

def generate_theme_preview(theme_details):
    w, h = 800, 450
    img = Image.new('RGB', (w, h), color=theme_details["background"])
    design_style = theme_details.get("design_style")
    
    if design_style in COMPOSITE_MAP:
        img = COMPOSITE_MAP[design_style](img, theme_details, w, h)

    draw = ImageDraw.Draw(img) if "RGBA" not in img.mode else ImageDraw.Draw(img, "RGBA")
    
    draw.text((50, 50), "Presentation Title", fill=theme_details["accent_color"], font=get_font(48))
    draw.text((70, 150), "• Bullet point one", fill=theme_details["font_color"], font=get_font(28))

    if design_style in PREVIEW_MAP:
        PREVIEW_MAP[design_style](draw, theme_details, w, h)
        
    return img.convert("RGB")