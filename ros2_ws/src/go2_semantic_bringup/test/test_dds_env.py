from go2_semantic_bringup.dds_env import DEPLOY_FRAGMENTS, cyclonedds_uri

UNITREE = '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="enP8p1s0"/></Interfaces></General></Domain></CycloneDDS>'


def test_appends_after_operator_uri():
    uri = cyclonedds_uri(UNITREE)
    assert uri.startswith(UNITREE + ",")
    assert all(f in uri for f in DEPLOY_FRAGMENTS)


def test_unset_or_blank_uri_has_no_leading_comma():
    for existing in (None, "", "  "):
        assert cyclonedds_uri(existing) == ",".join(DEPLOY_FRAGMENTS)


def test_idempotent_when_launched_from_an_env_that_already_has_it():
    once = cyclonedds_uri(UNITREE)
    assert cyclonedds_uri(once) == once


def test_multicast_limited_to_discovery():
    assert "<AllowMulticast>spdp</AllowMulticast>" in cyclonedds_uri(None)
