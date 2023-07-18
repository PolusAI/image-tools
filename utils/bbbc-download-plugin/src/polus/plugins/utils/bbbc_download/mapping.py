

def BBBC001_mapping(row: dict, extension: str) -> str:
    # important attributes: plate, well, wel num, control, field, channel, treatment, image type

    return f"a01_w01_n01_p01_f0{row['Field'] + 1}_c01_t00_i01{extension}"


__all__ = ["BBBC001_mapping"]