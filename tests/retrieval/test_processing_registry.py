from __future__ import annotations

import pytest

from skretrieval.retrieval.processing import Retrieval


def _resolve_optical_property(cls: type[Retrieval], name: str):
    # Retrieval looks optical properties up through the instance, construct one
    # without running __init__ so no observation or forward model is needed
    return object.__new__(cls)._optical_property(name)


@pytest.fixture
def instrument_retrievals():
    # Two instrument packages, e.g. SHOW and ALI, each subclassing Retrieval and
    # registering the same names
    class InstrumentA(Retrieval):
        pass

    class InstrumentB(Retrieval):
        pass

    @InstrumentA.register_optical_property("test_shared_aerosol")
    def a_optical_property():
        return "a"

    @InstrumentB.register_optical_property("test_shared_aerosol")
    def b_optical_property():
        return "b"

    @InstrumentA.register_state("aerosols", "test_shared_profile")
    def a_state(*_):
        return "a"

    @InstrumentB.register_state("aerosols", "test_shared_profile")
    def b_state(*_):
        return "b"

    return InstrumentA, InstrumentB


@pytest.fixture
def base_registration():
    name = "test_late_base_property"
    yield name
    Retrieval._optical_property_fns.pop(name, None)


def test_sibling_subclasses_keep_their_own_optical_properties(instrument_retrievals):
    a, b = instrument_retrievals

    assert _resolve_optical_property(a, "test_shared_aerosol") == "a"
    assert _resolve_optical_property(b, "test_shared_aerosol") == "b"


def test_sibling_subclasses_keep_their_own_states(instrument_retrievals):
    a, b = instrument_retrievals

    assert a._state_fns["aerosols"]["test_shared_profile"]() == "a"
    assert b._state_fns["aerosols"]["test_shared_profile"]() == "b"


@pytest.mark.usefixtures("instrument_retrievals")
def test_subclass_registrations_do_not_reach_the_base():
    assert "test_shared_aerosol" not in Retrieval._optical_property_fns
    assert "test_shared_profile" not in Retrieval._state_fns["aerosols"]


def test_subclass_resolves_the_default_registrations(instrument_retrievals):
    a, _ = instrument_retrievals

    assert a._optical_property_fns["o3"] is Retrieval._optical_property_fns["o3"]
    assert (
        a._state_fns["surface"]["lambertian_albedo"]
        is Retrieval._state_fns["surface"]["lambertian_albedo"]
    )


def test_subclass_resolves_base_registrations_made_after_it(
    instrument_retrievals, base_registration
):
    a, _ = instrument_retrievals

    Retrieval.register_optical_property(base_registration)(lambda: "base")

    assert _resolve_optical_property(a, base_registration) == "base"


def test_subclass_override_leaves_the_base_unchanged(instrument_retrievals):
    a, _ = instrument_retrievals
    base_o3 = Retrieval._optical_property_fns["o3"]

    a.register_optical_property("o3")(lambda: "a")

    assert _resolve_optical_property(a, "o3") == "a"
    assert Retrieval._optical_property_fns["o3"] is base_o3


def test_grandchild_resolves_through_its_parent(instrument_retrievals):
    a, _ = instrument_retrievals

    class Child(a):
        pass

    assert _resolve_optical_property(Child, "test_shared_aerosol") == "a"
    assert "o3" in Child._optical_property_fns
