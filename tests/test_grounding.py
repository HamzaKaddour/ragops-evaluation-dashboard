from ragops.grounding import is_abstention


def test_detects_insufficient_evidence_abstention():
    assert is_abstention("The available evidence is insufficient to answer this question.")


def test_normal_answer_is_not_abstention():
    assert not is_abstention("RAG retrieves external evidence before generation [doc_01].")
