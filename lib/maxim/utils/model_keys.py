

def translate_values(field,in_values):
    if field == "attn_mode":
        out_values = []
        for _v in in_values:
            v = translate_attn_mode(_v)
            out_values.append(v)
    elif field == "freeze":
        out_values = []
        for _v in in_values:
            v = translate_freeze(_v)
            out_values.append(v)
    else:
        out_values = [int(v) for v in in_values]
    return out_values

def translate_freeze(_v):
    return _v == "t"

def translate_attn_mode(_v): # keys under "augmented/lewin_ref.py"
    if _v == "pd":
        v = "product_dnls"
    elif _v == "ld":
        v = "l2_dnls"
    elif _v == "wd":
        v = "window_dnls"
    elif _v == "wr":
        v = "window_refactored"
    elif _v == "w":
        v = "window_default"
    else:
        raise ValueError(f"Uknown [attn_mode] type [{_v}]")
    return v

def expand_attn_mode(in_attn_mode,nblocks=5):
    if "_" in in_attn_mode:
        attn_modes = [in_attn_mode for _ in range(nblocks)]
    else:
        attn_modes = in_attn_mode.split("-")
        attn_modes = [translate_attn_mode(v) for v in attn_modes]
    assert len(attn_modes) == nblocks
    return attn_modes
