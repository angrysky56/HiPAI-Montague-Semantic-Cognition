# Copyright 2025 Google DeepMind
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Montague Semantic Engine

A simple lambda calculus evaluator and type-driven compositional semantic engine.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from hipai.models import Observation


@dataclass
class SemanticType:
    """Base class for all semantic types."""
    name: str

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, SemanticType):
            return False
        return self.name == other.name


@dataclass
class ComplexType(SemanticType):
    """Represents a complex semantic type, composed of a domain and codomain."""
    domain: SemanticType
    codomain: SemanticType

    def __init__(self, domain: SemanticType, codomain: SemanticType):
        super().__init__(name=f"<{domain},{codomain}>")
        self.domain = domain
        self.codomain = codomain

    def __eq__(self, other):
        if not isinstance(other, ComplexType):
            return False
        return self.domain == other.domain and self.codomain == other.codomain

# Base Types
TYPE_E = SemanticType("e") # Entity
TYPE_T = SemanticType("t") # Truth value
TYPE_S = SemanticType("s") # World/State


@dataclass
class LambdaExpression:
    """A semantic expression with a specific type and evaluated function."""
    expr_type: SemanticType
    func: Callable[[Any], Any]
    repr_str: str

    def apply(self, arg: 'LambdaExpression') -> 'LambdaExpression':
        """Applies this lambda expression to another expression as an argument."""
        func_type = self.expr_type
        if not isinstance(func_type, ComplexType):
            raise TypeError(f"Cannot apply non-function type {func_type}")

        func_type = cast(ComplexType, func_type)

        # Capture narrowed members to avoid type checker confusion after function calls
        domain = func_type.domain
        codomain = func_type.codomain

        if domain != arg.expr_type:
            raise TypeError(
                f"Type mismatch: expected {domain}, got {arg.expr_type}"
            )

        # Apply the function
        result_val = self.func(arg.func)
        return LambdaExpression(
            expr_type=codomain,
            # Handle both functional and non-functional returns
            func=result_val if callable(result_val) else lambda x=None: result_val,
            repr_str=f"({self.repr_str}({arg.repr_str}))"
        )

    def __str__(self):
        return f"{self.repr_str} : {self.expr_type}"

def lift(entity: LambdaExpression) -> LambdaExpression:
    """
    Type-shifting operator: Lift (Partee 1986).
    Shifts an entity of type e to a generalized quantifier of type ((e -> t) -> t).
    """
    if entity.expr_type != TYPE_E:
        raise TypeError("Can only lift expressions of type e.")

    gq_type = ComplexType(ComplexType(TYPE_E, TYPE_T), TYPE_T)
    # lambda P: P(entity)
    return LambdaExpression(
        expr_type=gq_type,
        func=lambda p_func: p_func(entity.func),
        repr_str=f"λP.P({entity.repr_str})"
    )

class SemanticEngine:
    """Engine for composing semantic expressions based on their types."""
    def compose_forward(
        self, func_expr: LambdaExpression, arg_expr: LambdaExpression
    ) -> LambdaExpression:
        """Forward application: func(arg)"""
        return func_expr.apply(arg_expr)

    def compose_backward(
        self, arg_expr: LambdaExpression, func_expr: LambdaExpression
    ) -> LambdaExpression:
        """Backward application: arg func -> func(arg)"""
        return func_expr.apply(arg_expr)

    def apply_type_driven(
        self, expr1: LambdaExpression, expr2: LambdaExpression
    ) -> LambdaExpression:
        """Attempts to compose two expressions based on their types."""
        type1 = expr1.expr_type
        type2 = expr2.expr_type

        # Try forward: expr1(expr2)
        if (
            isinstance(type1, ComplexType)
            and type1.domain == type2
        ):
            return self.compose_forward(expr1, expr2)

        # Try backward: expr2(expr1)
        if (
            isinstance(type2, ComplexType)
            and type2.domain == type1
        ):
            return self.compose_backward(expr1, expr2)

        raise TypeError(f"Cannot compose {type1} and {type2}")

    def compile_observation(self, observation: "Observation") -> list[LambdaExpression]:
        """
        Translates an Observation's relations and individuals into LambdaExpressions
        and type-checks them by applying the relations to the individuals.
        Returns the resulting TYPE_T expressions.
        """
        results = []
        ind_map = {}

        # Compile individuals to type e
        for ind in observation.individuals:
            ind_expr = LambdaExpression(
                expr_type=TYPE_E,
                func=lambda x, name=ind.name: name,
                repr_str=ind.name
            )
            ind_map[ind.id] = ind_expr

        # Compile relations to type <e, <e, t>>
        for rel in observation.relations:
            rel_type = ComplexType(TYPE_E, ComplexType(TYPE_E, TYPE_T))

            # The function takes an object y (type e), then a subject x (type e),
            # and returns True (type t)
            rel_expr = LambdaExpression(
                expr_type=rel_type,
                func=lambda y: lambda x: True,
                repr_str=f"λy.λx.{rel.relation_type}(x, y)"
            )

            subj_expr = ind_map.get(rel.source_id)
            obj_expr = ind_map.get(rel.target_id)

            if subj_expr and obj_expr:
                # Apply relation to object first: rel_expr(obj_expr) -> <e, t>
                pred_expr = self.compose_forward(rel_expr, obj_expr)
                # Then apply result to subject: pred_expr(subj_expr) -> t
                prop_expr = self.compose_forward(pred_expr, subj_expr)
                results.append(prop_expr)

        return results
