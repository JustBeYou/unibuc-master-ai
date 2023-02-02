debug :- fail.

:- initialization(main).
main :- 
    load_kb('kb1.txt', Rules, Questions, Goal),
    writeln("Rules: "), maplist(writeln, Rules), nl,
    write("Goal: "), writeln(Goal), nl,
    repeat,
    maplist(ask_question, Questions, Answers),
    append(Rules, Answers, KB),
    nl,
    write("Answers: "), writeln(Answers), nl,
    write("Goal entailed (backward chaining): "),
    (
        (debug, nl; true),
        backward_chaining(KB, [Goal]) -> writeln("YES");
        writeln("NO")
    ),
    write("Goal entailed (forward chaining): "),
    (
        (debug, nl; true),
        forward_chaining(KB, [Goal], Answers) -> writeln("YES");
        writeln("NO")
    ),
    nl,
    fail.

backward_chaining(_, []).
backward_chaining(KB, [Goal | RestGoals]) :- 
    (debug, write("[DEBUG] Goal = "), write(Goal), write(" RestGoals = "), writeln(RestGoals); true),
    member(Clause, KB),
    member(Goal, Clause),
    select(Goal, Clause, Premises),
    maplist(negate, Premises, NegPremises),
    append(NegPremises, RestGoals, NewGoals),
    backward_chaining(KB, NewGoals), !.

forward_chaining(_, Goals, Solved) :- subset(Goals, Solved).
forward_chaining(KB, Goals, Solved) :- 
    (debug, write("[DEBUG] Goals ="), write(Goals), write(" Solved = "), writeln(Solved); true),
    member(Clause, KB),
    member(Goal, Clause),
    is_positive(Goal),
    \+ member(Goal, Solved),
    select(Goal, Clause, Premises),
    maplist(negate, Premises, NegPremises),
    subset(NegPremises, Solved),
    append(Solved, [Goal], NewSolved),
    forward_chaining(KB, Goals, NewSolved).

subset([], _).
subset(A, B) :- maplist(rev_member(B), A).
rev_member(L, M) :- member(M, L). 

:- dynamic rules/1.
:- dynamic goal/1.
:- dynamic questions/1.
load_kb(Filename, Rules, Questions, Goal) :-
	see(Filename),
    repeat,
        read(Term),
        ( Term == end_of_file; assertz(Term), fail),
	seen,
    rules(Rules),
    questions(Questions),
    goal(Goal).

ask_question([Question, Term], [Answer]) :-
    write(Question), write(" (term: "), write(Term), write(") "), write("> "), 
    read_line_to_string(user_input, Input),
    (
        member(Input, ["y", "yes"]) -> Answer = Term;
        member(Input, ["n", "no"]) -> (negate(Term, Neg_Term), Answer = Neg_Term);
        Input = "stop" -> halt;
        writeln("Invalid answer. Must be yes or no."), halt
    ).

is_positive(n(_)) :- !, fail.
is_positive(_).

negate(n(P), P) :- !.
negate(P, n(P)).