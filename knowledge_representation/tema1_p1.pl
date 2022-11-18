:- initialization(main).
main :- 
	load_testcase('testdata/both_clauses_equal_after_removing_resolvent.txt', "not satisfied", _),
	load_testcase('testdata/clauses_different_after_removing_resolvent.txt', "not satisfied", _),
	load_testcase('testdata/satisfied.txt', "satisfied", _),
	
	load_testcase('testdata/assignment_d_i.txt', "not satisfied", _),
	load_testcase('testdata/assignment_d_ii.txt', "not satisfied", _),
	load_testcase('testdata/assignment_d_iii.txt', "satisfied", _),
	load_testcase('testdata/assignment_d_iv.txt', "satisfied", _),
	
	writeln(""),
	load_testcase('testdata/my_assignment_example.txt', "not satisfied", _),
	writeln("The clauses in my_assignment_example.txt contain the negation of the proposition to be proved."),
	writeln("Because the clauses generated the empty clause (system not satisfiable), we consider the proposition to be entitled from the KB."),
	
	halt.

:- dynamic test_case_KB/1.
load_testcase(Filename, Expected, Result) :-
	see(Filename),
	read(TestCase_KB),
	assertz(TestCase_KB),
	test_case_KB(KB),
	include(not_tautology, KB, Filtered_KB),
	test_case(Filename, Filtered_KB, Expected, Result),
	retractall(test_case_KB(_)),
	seen.
	
test_case(TestCase, KB, Expected, Result) :-
	write("Test "), write(TestCase), write(", "), write(Expected), write(" expected => result: "),
	( (resolution(KB), Result = "not satisfied") ; Result = "satisfied" ),
	write(Result),
	( Result == Expected -> writeln(" [OK]") ; writeln(" [FAILED]") ).

resolution(KB) :- member([], KB), !.
resolution(KB) :- 
    member(Premise_1, KB),
    member(Premise_2, KB),
	Premise_1 \= Premise_2,
    resolve(Conclusion, Premise_1, Premise_2),
	\+ member(Conclusion, KB),
	\+ tautology(Conclusion),
    resolution([Conclusion | KB]), !.

tautology(Clause) :- member(P, Clause), negate(P, Not_P), member(Candidate_Not_P, Clause), subsumes_term(Not_P, Candidate_Not_P).
not_tautology(Clause) :- \+ tautology(Clause).

resolve(Conclusion, Premise_1, Premise_2) :- 
    select(Resolvent, Premise_1, Premise_1_prime),
	negate(Resolvent, Not_Resolvent),
    select(Not_Resolvent, Premise_2, Premise_2_prime),
    append(Premise_1_prime, Premise_2_prime, Conclusion_prime),
    list_to_set(Conclusion_prime, Conclusion).

negate(not(P), P) :- !.
negate(P, not(P)).
