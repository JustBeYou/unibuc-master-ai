:- initialization(main).
main :- 
	% load_testcase('testdata/both_clauses_equal_after_removing_resolvent.txt', "NO"),
	% load_testcase('testdata/clauses_different_after_removing_resolvent.txt', "NO"),
	% load_testcase('testdata/satisfied.txt', "YES"),
	
	% load_testcase('testdata/assignment_d_i.txt', "NO"),
	% load_testcase('testdata/assignment_d_ii.txt', "NO"),
	load_testcase('testdata/assignment_d_iii.txt', "YES"),
	% load_testcase('testdata/assignment_d_iv.txt', "YES"),
	
	% load_testcase('testdata/my_assignment_example.txt', "NO"),
	
	halt.

default_strategy(first_atom).

:- dynamic test_case_KB/1.
load_testcase(Filename, Expected) :-
	see(Filename),
	read(TestCase_KB),
	assertz(TestCase_KB),
	test_case_KB(KB),
	test_case(Filename, KB, Expected),
	retractall(test_case_KB(_)),
	seen.
	
test_case(TestCase, KB, Expected) :-
    default_strategy(Strategy),
    dp(Strategy, KB, Result),
	write("Test "), write(TestCase), write(", "), write(Expected), write(" expected => result: "),
	( Result == Expected -> writeln(" [OK]") ; writeln(" [FAILED]") ).

first_atom(Atom, Atoms) :- member(Atom, Atoms).

dp(_, [], "YES").
dp(_, Clauses, "NO") :- member([], Clauses).
dp(Strategy, Clauses, Satisfiable) :-
    flatten(Clauses, AtomsDuplicates),
    list_to_set(AtomsDuplicates, Atoms),
    call(Strategy, Atom, Atoms),
    write("atom: "), writeln(Atom),
    (
        (
            unit_propagation(Clauses, Atom, PositiveResult),
            write("positive branch: "), writeln(PositiveResult),
            dp(Strategy, PositiveResult, Satisfiable)
        ) ;
        (
            unit_propagation(Clauses, negate(Atom), NegativeResult),
            write("negative branch: "), writeln(NegativeResult),
            dp(Strategy, NegativeResult, Satisfiable)
        )
    ).


unit_propagation(Clauses, Atom, Result) :-
    positive_part(Clauses, Atom, PositiveResult),
    negative_part(Clauses, Atom, NegativeResult),
    append(PositiveResult, NegativeResult, ResultDuplicates),
    list_to_set(ResultDuplicates, Result).

positive_part([], _, []).
positive_part([Clause | OtherClauses], Atom, [Clause | Rest]) :- 
    \+ member(Atom, Clause),
    negate(Atom, Not_Atom),
    \+ member(Not_Atom, Clause),
    positive_part(OtherClauses, Atom, Rest), !.
positive_part([_ | OtherClauses], Atom, Result) :- positive_part(OtherClauses, Atom, Result), !.

negative_part([], _, []).
negative_part([Clause | OtherClauses], Atom, [NewClause | Rest]) :-
    \+ member(Atom, Clause),
    negate(Atom, Not_Atom),
    member(Not_Atom, Clause),
    select(Not_Atom, Clause, NewClause),
    negative_part(OtherClauses, Atom, Rest).
negative_part([_ | OtherClauses], Atom, Result) :- negative_part(OtherClauses, Atom, Result), !.

negate(not(P), P) :- !.
negate(P, not(P)).
