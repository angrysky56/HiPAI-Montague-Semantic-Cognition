import pytest

from hipai.semantics import (
    TYPE_E,
    TYPE_T,
    ComplexType,
    LambdaExpression,
    SemanticEngine,
)


def test_semantic_types():
    assert str(TYPE_E) == "e"
    assert str(TYPE_T) == "t"
    
    type_et = ComplexType(TYPE_E, TYPE_T)
    assert str(type_et) == "<e,t>"
    
    type_e_et = ComplexType(TYPE_E, type_et)
    assert str(type_e_et) == "<e,<e,t>>"
    
    assert type_et == ComplexType(TYPE_E, TYPE_T)
    assert type_et != type_e_et

def test_lambda_application():
    # Let's create an entity "john"
    john = LambdaExpression(expr_type=TYPE_E, func=lambda x: "John", repr_str="j")
    
    # And a property "runs"
    runs = LambdaExpression(
        expr_type=ComplexType(TYPE_E, TYPE_T), 
        func=lambda x: f"run({x(None)})", 
        repr_str="run"
    )
    
    # Apply forward
    engine = SemanticEngine()
    result = engine.compose_forward(runs, john)
    
    assert result.expr_type == TYPE_T
    assert result.repr_str == "(run(j))"
    assert result.func() == "run(John)"

def test_type_driven_composition():
    john = LambdaExpression(expr_type=TYPE_E, func=lambda x: "John", repr_str="j")
    runs = LambdaExpression(
        expr_type=ComplexType(TYPE_E, TYPE_T), 
        func=lambda x: f"run({x(None)})", 
        repr_str="run"
    )
    
    engine = SemanticEngine()
    
    # Try forward application: runs(john)
    res1 = engine.apply_type_driven(runs, john)
    assert res1.repr_str == "(run(j))"
    
    # Try backward application: john runs -> runs(john)
    res2 = engine.apply_type_driven(john, runs)
    assert res2.repr_str == "(run(j))"
    
    # Try invalid composition
    with pytest.raises(TypeError):
        engine.apply_type_driven(john, john)
