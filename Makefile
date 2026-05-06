ISABELLE_BIN ?= isabelle

.PHONY: verify-logic

verify-logic:
	$(ISABELLE_BIN) build -D docs/logic
