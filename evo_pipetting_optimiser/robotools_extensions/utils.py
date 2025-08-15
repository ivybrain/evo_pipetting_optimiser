def check_offset_limits(op, tip):
    source_limit_up = getattr(op.source, "offset_limit_up", None)
    source_limit_down = getattr(op.source, "offset_limit_down", None)
    dest_limit_up = op.destination.offset_limit_up
    dest_limit_down = op.destination.offset_limit_down

    source_offset = op.source_pos[0] - tip
    dest_offset = op.dest_pos[0] - tip

    def check_limit(offset, limit, subtract=False):
        # Check if we have exceeded the offset limit allowed
        if limit is None:
            return 0
        if subtract:
            return offset - limit
        return offset + limit

    source_up_check = check_limit(source_offset, source_limit_up)
    source_down_check = check_limit(source_offset, source_limit_down, True)
    dest_up_check = check_limit(dest_offset, dest_limit_up)
    dest_down_check = check_limit(dest_offset, dest_limit_down, True)

    return min(source_up_check, dest_up_check), max(source_down_check, dest_down_check)
