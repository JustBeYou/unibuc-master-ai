:- initialization(main).
main :- 
    load_testcase('testdata/clauses_different_after_removing_resolvent.txt', "NO"),
    load_testcase('testdata/satisfied.txt', "YES"),

    load_testcase('testdata/assignment_d_i.txt', "NO"),
    load_testcase('testdata/assignment_d_ii.txt', "NO"),
    load_testcase('testdata/assignment_d_iii.txt', "YES"),
    load_testcase('testdata/assignment_d_iv.txt', "YES"),

    load_testcase('testdata/assignment_2_i.txt', "YES"),
    load_testcase('testdata/assignment_2_ii.txt', "NO"), 
    load_testcase('testdata/assignment_2_iii.txt', "NO"),
    load_testcase('testdata/assignment_2_iv.txt', "NO"),
    load_testcase('testdata/assignment_2_v.txt', "YES"),
    load_testcase('testdata/assignment_2_vi.txt', "NO"),

    load_testcase('testdata/my_assignment_example.txt', "NO"),

	halt.

:- dynamic test_case_KB/1.
load_testcase(Filename, Expected) :-
	see(Filename),
	read(TestCase_KB),
	seen,
	assertz(TestCase_KB),
    test_case_KB(KB),
	retractall(test_case_KB(_)),
    test_case(Filename, KB, Expected).
	
test_case(TestCase, KB, Expected) :-
    ( dp(KB, Solution) -> Result = "YES" ; Result = "NO" ),
	write("Test "), write(TestCase), write(", "), write(Expected), write(" expected => result: "), write(Result),
	( Result == Expected -> writeln(" [OK]") ; writeln(" [FAILED]") ),
    ( Result == "YES" -> 
        writeln("Solution available: "), 
        display_solution(Solution); true).

dp([], []) :- !.
dp(Clauses, []) :- member([], Clauses), !, fail.
dp(Clauses, [Atom | Rest]) :-
    select_atom(Atom, Clauses),
    ( 
        (
            apply_atom(Clauses, Atom, AffirmativeResult),
            dp(AffirmativeResult, Rest)
        ) ;
        (
            negate(Atom, Not_Atom),
            apply_atom(Clauses, Not_Atom, NegativeResult),
            dp(NegativeResult, Rest)
        )
    ).

display_solution([]) :- writeln(""), writeln("The rest of the variables can have any value.").
display_solution([P | Rest]) :- 
    substitute_solution_predicate(P, Output_P, Value),
    write(Output_P), write("/"), write(Value), write(" "),
    display_solution(Rest).

first_atom(Atom, [FristClause | _]) :- member(Atom, FristClause), !.

atom_of_shortest_clause(Atom, Clauses) :-
	maplist(length, Clauses, Lengths),
    min_list(Lengths, MinLength),
    member(ShortestClause, Clauses),
    length(ShortestClause, ClauseLength),
    ClauseLength == MinLength, 
    member(Atom, ShortestClause), !.

most_frequent_atom(Atom, Clauses) :-
    	flatten(Clauses, Atoms),
        aggregate(
            max(Count,Element),
        	aggregate(
                count,
                member(Element,Atoms),Count
            ), max(_, Atom)
        ), !.

select_atom(Atom, Clauses) :- most_frequent_atom(Atom, Clauses), !.

apply_atom(Clauses, Atom, Result) :-
    affirmative_rule(Clauses, Atom, AffirmativeResult),
    negative_rule(Clauses, Atom, NegativeResult),
    append(AffirmativeResult, NegativeResult, ResultDuplicates),
    list_to_set(ResultDuplicates, Result), !.

affirmative_rule([], _, []).
affirmative_rule([Clause | OtherClauses], Atom, [Clause | Rest]) :- 
    \+ member(Atom, Clause),
    negate(Atom, Not_Atom), 
    \+ member(Not_Atom, Clause),
    affirmative_rule(OtherClauses, Atom, Rest), !.
affirmative_rule([_ | OtherClauses], Atom, Result) :- affirmative_rule(OtherClauses, Atom, Result), !.

negative_rule([], _, []).
negative_rule([Clause | OtherClauses], Atom, [NewClause | Rest]) :-
    \+ member(Atom, Clause),
    negate(Atom, Not_Atom),
    member(Not_Atom, Clause), 
    select(Not_Atom, Clause, NewClause),
    negative_rule(OtherClauses, Atom, Rest), !.
negative_rule([_ | OtherClauses], Atom, Result) :- negative_rule(OtherClauses, Atom, Result), !.



substitute_solution_predicate(not(P), P, false) :- !.
substitute_solution_predicate(P, P, true).

negate(not(P), P) :- !.
negate(P, not(P)).