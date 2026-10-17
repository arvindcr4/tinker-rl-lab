"""Pure retained-evidence checks; labels never substitute for observed counts."""


def is_graded(status):
    return status in ("GRADED", "GRADED_OK")


def components_match(components, total):
    return (
        isinstance(components, dict)
        and bool(components)
        and type(total) is int
        and total >= 0
        and all(
            isinstance(c, dict) and type(c.get("passes")) is int and c["passes"] >= 0
            for c in components.values()
        )
        and sum(c["passes"] for c in components.values()) == total
    )


def verification_exit_code(checks):
    return 1 if not checks or any(c.get("status") != "PASS" for c in checks) else 0
