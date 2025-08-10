#!/usr/bin/env python3
from pathlib import Path

# --- settings ---
OUT_DIR = Path("assets/icons")
SKIP_EXISTING = True  # set to False if you want to overwrite any existing SVGs
STROKE = "currentColor"
STROKE_W = "1.8"

def svg(title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'width="24" height="24" fill="none" stroke="{STROKE}" '
        f'stroke-width="{STROKE_W}" stroke-linecap="round" stroke-linejoin="round">'
        f'<title>{title}</title>{body}</svg>\n'
    )

# --- simple drawing primitives (minimal placeholders) ---

def icon_default(t):  # circle + dot
    return svg(t, '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="1.5"/>')

def icon_leaf(t):
    return svg(t, '<path d="M4 14c5-8 10-8 16-4-2 6-8 10-14 8l3-3"/><path d="M8 18c1-3 4-6 9-8"/>')

def icon_recycle(t):
    return svg(t,
      '<path d="M7 7l2.2-3.5L12 6"/><path d="M10 6H6"/>'  # small arrow
      '<path d="M17 7l-2.2-3.5L12 6"/><path d="M14 6h4"/>' # small arrow
      '<path d="M7 17l2.2 3.5L12 18"/><path d="M10 18H6"/>' # small arrow
    )

def icon_sun(t):
    rays = ''.join([f'<line x1="12" y1="2" x2="12" y2="5"/>',
                    f'<line x1="12" y1="19" x2="12" y2="22"/>',
                    f'<line x1="2" y1="12" x2="5" y2="12"/>',
                    f'<line x1="19" y1="12" x2="22" y2="12"/>',
                    f'<line x1="4.2" y1="4.2" x2="6.3" y2="6.3"/>',
                    f'<line x1="17.7" y1="17.7" x2="19.8" y2="19.8"/>',
                    f'<line x1="17.7" y1="6.3" x2="19.8" y2="4.2"/>',
                    f'<line x1="4.2" y1="19.8" x2="6.3" y2="17.7"/>'])
    return svg(t, '<circle cx="12" cy="12" r="4"/>' + rays)

def icon_wind(t):
    return svg(t, '<path d="M3 9h9a3 3 0 100-6"/><path d="M3 15h13a3 3 0 110 6"/>')

def icon_water(t):
    return svg(t, '<path d="M12 3c4 5 6 8 6 11a6 6 0 11-12 0c0-3 2-6 6-11z"/>')

def icon_flame(t):
    return svg(t, '<path d="M12 3c3 4 1 5 3 7a4.5 4.5 0 11-9 1c0-2 1-3 3-5 1 2 3 3 3-3z"/>')

def icon_plug(t):
    return svg(t, '<path d="M6 9v4a5 5 0 005 5h2a5 5 0 005-5V9"/><line x1="9" y1="2" x2="9" y2="6"/><line x1="15" y1="2" x2="15" y2="6"/>')

def icon_heart(t):
    return svg(t, '<path d="M20.8 8.6a5.5 5.5 0 00-9-1.7 5.5 5.5 0 00-9 1.7C2.8 13 12 20 12 20s9.2-7 8.8-11.4z"/>')

def icon_stethoscope(t):
    return svg(t, '<path d="M6 3v4a4 4 0 008 0V3"/><path d="M10 11a4 4 0 108 0v2a5 5 0 01-5 5h-1"/><circle cx="18" cy="5" r="2"/>')

def icon_pill(t):
    return svg(t, '<rect x="3" y="8" width="10" height="8" rx="4"/><rect x="11" y="8" width="10" height="8" rx="4"/><line x1="8" y1="8" x2="8" y2="16"/>')

def icon_dna(t):
    return svg(t, '<path d="M7 3c5 3 5 6 0 9"/><path d="M17 3c-5 3-5 6 0 9"/><path d="M7 12c5 3 5 6 0 9"/><path d="M17 12c-5 3-5 6 0 9"/>')

def icon_wheelchair(t):
    return svg(t, '<circle cx="12" cy="5" r="2"/><path d="M12 7v5h5"/><circle cx="10" cy="16" r="4"/><path d="M10 12l3 6"/>')

def icon_baby(t):
    return svg(t, '<circle cx="12" cy="8" r="3"/><path d="M8 15a4 4 0 008 0"/><circle cx="9" cy="8.5" r="0.6"/><circle cx="15" cy="8.5" r="0.6"/>')

def icon_bank(t):
    return svg(t, '<path d="M4 9h16l-8-4z"/><path d="M5 10v7M9 10v7M15 10v7M19 10v7"/><path d="M3 17h18"/>')

def icon_credit_card(t):
    return svg(t, '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M3 10h18"/><path d="M7 14h4"/>')

def icon_chart(t):
    return svg(t, '<path d="M3 19h18"/><rect x="5" y="11" width="3" height="6"/><rect x="10" y="7" width="3" height="10"/><rect x="15" y="9" width="3" height="8"/>')

def icon_shield_dollar(t):
    return svg(t, '<path d="M12 21c6-3 7-6 7-10V7l-7-3-7 3v4c0 4 1 7 7 10"/><path d="M10 11c0-1 1-2 2-2s2 .6 2 1.5-1 1.5-2 1.5-2 .6-2 1.5 1 1.5 2 1.5 2-1 2-2"/><path d="M12 9v6"/>')

def icon_courthouse(t):
    return svg(t, '<path d="M3 10h18l-9-5z"/><path d="M5 10v7M9 10v7M15 10v7M19 10v7"/><path d="M3 17h18"/>')

def icon_scale(t):
    return svg(t, '<path d="M12 3v18"/><path d="M5 7l-3 6h6l-3-6z"/><path d="M19 7l-3 6h6l-3-6z"/>')

def icon_id_card(t):
    return svg(t, '<rect x="3" y="6" width="18" height="12" rx="2"/><circle cx="8.5" cy="12" r="2"/><path d="M13 11h6M13 14h6"/>')

def icon_ballot(t):
    return svg(t, '<rect x="4" y="6" width="16" height="12" rx="2"/><path d="M8 10l2-2 2 2-2 2-2-2z"/><path d="M14 14h4"/>')

def icon_graduation(t):
    return svg(t, '<path d="M2 9l10-5 10 5-10 5-10-5z"/><path d="M4 10v4c3 2 11 2 14 0v-4"/>')

def icon_book(t):
    return svg(t, '<path d="M5 4h11a3 3 0 013 3v13H8a3 3 0 00-3 3z"/><path d="M5 4v16a3 3 0 013-3h11"/>')

def icon_museum(t):
    return svg(t, '<path d="M3 9h18l-9-5z"/><path d="M5 10v7M9 10v7M15 10v7M19 10v7"/><path d="M3 17h18"/>')

def icon_house(t):
    return svg(t, '<path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/>')

def icon_blueprint(t):
    return svg(t, '<rect x="5" y="5" width="12" height="12" rx="2"/><path d="M9 5v12M5 9h12"/>')

def icon_construction(t):
    return svg(t, '<path d="M4 20h16"/><path d="M7 20l4-10 4 10"/><path d="M9 15h6"/>')

def icon_wrench(t):
    return svg(t, '<path d="M14 4a4 4 0 11-4 4L4 14l3 3 6-6"/>')

def icon_key(t):
    return svg(t, '<circle cx="9" cy="9" r="3"/><path d="M12 9h9l-2 2-2-2-2 2-2-2"/>')

def icon_cart(t):
    return svg(t, '<path d="M4 6h2l2 10h9l2-6H8"/><circle cx="9" cy="19" r="1.5"/><circle cx="17" cy="19" r="1.5"/>')

def icon_tag(t):
    return svg(t, '<path d="M20 10l-8 8-8-8V4h6z"/><circle cx="9" cy="7" r="1.2"/>')

def icon_fork_knife(t):
    return svg(t, '<path d="M6 3v8M6 11v10"/><path d="M10 3v5a2 2 0 002 2h0v11"/>')

def icon_bed(t):
    return svg(t, '<path d="M3 18V9h8a6 6 0 016 6v3"/><path d="M3 14h18"/><path d="M3 18h18"/>')

def icon_plane(t):
    return svg(t, '<path d="M2 12l20-7-6 7 6 7-20-7z"/>')

def icon_truck(t):
    return svg(t, '<rect x="3" y="7" width="10" height="8"/><path d="M13 11h4l3 3v1h-7z"/><circle cx="7" cy="17" r="1.5"/><circle cx="17" cy="17" r="1.5"/>')

def icon_car(t):
    return svg(t, '<path d="M5 16l1-4 3-2h6l3 2 1 4"/><path d="M4 16h16"/><circle cx="8" cy="17.5" r="1.5"/><circle cx="16" cy="17.5" r="1.5"/>')

def icon_bus(t):
    return svg(t, '<rect x="4" y="5" width="16" height="10" rx="2"/><path d="M4 11h16"/><circle cx="8" cy="17" r="1"/><circle cx="16" cy="17" r="1"/>')

def icon_code(t):
    return svg(t, '<path d="M8 6l-4 6 4 6"/><path d="M16 6l4 6-4 6"/>')

def icon_cloud(t):
    return svg(t, '<path d="M5 17h12a4 4 0 100-8 6 6 0 10-12 2"/>')

def icon_database(t):
    return svg(t, '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v8c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/><path d="M5 10c0 1.7 3.1 3 7 3s7-1.3 7-3"/>')

def icon_brain(t):
    return svg(t, '<path d="M9 8a3 3 0 10-3 3v2a3 3 0 103 3"/><path d="M15 8a3 3 0 013-3 3 3 0 010 6v2a3 3 0 11-3 3"/>')

def icon_server(t):
    return svg(t, '<rect x="4" y="5" width="16" height="5" rx="2"/><rect x="4" y="14" width="16" height="5" rx="2"/><circle cx="7" cy="7.5" r="0.8"/><circle cx="7" cy="16.5" r="0.8"/>')

def icon_shield(t):
    return svg(t, '<path d="M12 21c6-3 7-6 7-10V7l-7-3-7 3v4c0 4 1 7 7 10"/>')

def icon_bug(t):
    return svg(t, '<circle cx="12" cy="10" r="3"/><path d="M5 10h14"/><path d="M7 7l-2-2M17 7l2-2"/><path d="M7 13l-2 2M17 13l2 2"/><path d="M12 13v6"/>')

def icon_camera(t):
    return svg(t, '<path d="M4 7h4l2-2h4l2 2h4v10H4z"/><circle cx="12" cy="12" r="3"/>')

def icon_music(t):
    return svg(t, '<path d="M9 18a2 2 0 100-4 2 2 0 000 4z"/><path d="M15 16a2 2 0 100-4 2 2 0 000 4z"/><path d="M11 14V6l8-2v8"/>')

def icon_trophy(t):
    return svg(t, '<path d="M8 21h8"/><path d="M10 21v-3h4v3"/><path d="M6 5h12v4a6 6 0 01-12 0z"/><path d="M4 7h2M18 7h2"/>')

def icon_gamepad(t):
    return svg(t, '<rect x="4" y="9" width="16" height="8" rx="3"/><path d="M8 13h4M10 11v4"/><circle cx="16" cy="12" r="0.8"/><circle cx="18" cy="14" r="0.8"/>')

def icon_wheat(t):
    return svg(t, '<path d="M12 3v18"/><path d="M12 7l-3-2M12 7l3-2M12 11l-3-2M12 11l3-2M12 15l-3-2M12 15l3-2"/>')

def icon_factory(t):
    return svg(t, '<path d="M3 20h18V9l-5 3V9l-5 3V9H3z"/><rect x="6" y="14" width="3" height="3"/>')

def icon_pickaxe(t):
    return svg(t, '<path d="M3 7h9l2 2 7-2"/><path d="M12 7v14"/>')

def icon_users(t):
    return svg(t, '<circle cx="9" cy="8" r="3"/><circle cx="16" cy="9" r="2.5"/><path d="M3 19a6 6 0 0112 0"/><path d="M12 19a5 5 0 019 0"/>')

def icon_handshake(t):
    return svg(t, '<path d="M7 12l3 3a2 2 0 003 0l4-4"/><path d="M3 12l4-4 3 3M21 12l-4-4-3 3"/>')

def icon_charity(t):
    return svg(t, '<path d="M20 9c0 6-8 10-8 10S4 15 4 9a4 4 0 018-2 4 4 0 018 2z"/>')

def icon_briefcase(t):
    return svg(t, '<rect x="3" y="7" width="18" height="12" rx="2"/><path d="M9 7V5h6v2"/><path d="M3 12h18"/>')

def icon_bolt(t):
    return svg(t, '<path d="M13 2L4 14h6l-1 8 9-12h-6l1-8z"/>')

def icon_droplet(t):
    return svg(t, '<path d="M12 3c4 5 6 8 6 11a6 6 0 11-12 0c0-3 2-6 6-11z"/>')

def icon_cloud_sun(t):
    return svg(t, '<circle cx="6" cy="6" r="2.5"/><path d="M5 17h12a4 4 0 100-8 6 6 0 10-12 2"/>')

TEMPLATES = {
    "leaf": icon_leaf, "recycle": icon_recycle, "sun": icon_sun, "wind": icon_wind,
    "water": icon_water, "flame": icon_flame, "plug": icon_plug, "heart": icon_heart,
    "stethoscope": icon_stethoscope, "pill": icon_pill, "dna": icon_dna, "wheelchair": icon_wheelchair,
    "baby": icon_baby, "bank": icon_bank, "credit-card": icon_credit_card, "chart": icon_chart,
    "shield-dollar": icon_shield_dollar, "courthouse": icon_courthouse, "scale": icon_scale,
    "id-card": icon_id_card, "ballot": icon_ballot, "graduation": icon_graduation, "book": icon_book,
    "museum": icon_museum, "house": icon_house, "blueprint": icon_blueprint, "construction": icon_construction,
    "wrench": icon_wrench, "key": icon_key, "cart": icon_cart, "tag": icon_tag, "fork-knife": icon_fork_knife,
    "bed": icon_bed, "plane": icon_plane, "truck": icon_truck, "car": icon_car, "bus": icon_bus,
    "code": icon_code, "cloud": icon_cloud, "database": icon_database, "brain": icon_brain,
    "server": icon_server, "shield": icon_shield, "bug": icon_bug, "camera": icon_camera,
    "music": icon_music, "trophy": icon_trophy, "gamepad": icon_gamepad, "wheat": icon_wheat,
    "factory": icon_factory, "pickaxe": icon_pickaxe, "users": icon_users, "handshake": icon_handshake,
    "charity": icon_charity, "briefcase": icon_briefcase, "bolt": icon_bolt, "droplet": icon_droplet,
    "cloud-sun": icon_cloud_sun, "default": icon_default,
}

ALL_FILES = [
    "leaf.svg","recycle.svg","sun.svg","wind.svg","water.svg","flame.svg","plug.svg",
    "heart.svg","stethoscope.svg","pill.svg","dna.svg","wheelchair.svg","baby.svg",
    "bank.svg","credit-card.svg","chart.svg","shield-dollar.svg","courthouse.svg","scale.svg",
    "id-card.svg","ballot.svg","graduation.svg","book.svg","museum.svg","house.svg","blueprint.svg",
    "construction.svg","wrench.svg","key.svg","cart.svg","tag.svg","fork-knife.svg","bed.svg",
    "plane.svg","truck.svg","car.svg","bus.svg","code.svg","cloud.svg","database.svg","brain.svg",
    "server.svg","shield.svg","bug.svg","camera.svg","music.svg","trophy.svg","gamepad.svg","wheat.svg",
    "factory.svg","pickaxe.svg","users.svg","handshake.svg","charity.svg","briefcase.svg","bolt.svg",
    "droplet.svg","cloud-sun.svg","default.svg"
]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = 0
    for fname in ALL_FILES:
        stem = fname.replace(".svg", "")
        out = OUT_DIR / fname
        if SKIP_EXISTING and out.exists():
            continue
        func = TEMPLATES.get(stem, icon_default)
        out.write_text(func(stem.replace("-", " ")), encoding="utf-8")
        created += 1
    print(f"Icons written to {OUT_DIR} (created/updated: {created}/{len(ALL_FILES)})")

if __name__ == "__main__":
    main()
