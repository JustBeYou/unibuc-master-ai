
:- op(498, xfy, [or, and]).
:- op(499, xfy, implies).

debug.

:- initialization(main).
main :- 
    load_kb('kb_lab.txt'),
    inputs(Inputs),
    (debug, show_curves; true),  
    repeat,
    maplist(ask_rating, Inputs), nl,

    findall(rule(R), rule(R), Rules),
    maplist(eval, Rules, _),

    output(Goal, MinValue, MaxValue, Precision),
    find_centroid(aggregate_degree, MinValue, MaxValue, Precision, Result),
    write("Estimated output for "), write(Goal), write(": "), writeln(Result),

    retractall(max_satisfaction(_, _)),
    retractall(rating(_, _)), nl,
    fail.

find_centroid(F, Left, Right, Delta, Result) :- 
    integrate(x_f(F), Left, Right, Delta, Result_X_F),
    integrate(F, Left, Right, Delta, Result_F),
    Result is Result_X_F / Result_F.

x_f(F, X, Y) :- call(F, X, Y0), Y is X * Y0.

integrate(F, Left, Right, Delta, Result) :- 
    NewRight is Right - Delta,
    range(Left, Delta, NewRight, Xs),
    maplist(add_half(Delta), Xs, MidXs),
    maplist(call(F), MidXs, Ys),
    maplist(multiply(Delta), Ys, Areas),
    sumlist(Areas, Result).

add_half(B, A, C) :- C is A + 1 / 2 * B.
multiply(B, A, C) :- C is B * A.

range(Start, _, Stop, []) :- Start >= Stop, !.
range(Start, Step, Stop, [Start | Other]) :-
    Start =< Stop,
    Next is Start + Step,
    range(Next, Step, Stop, Other).

aggregate_degree(X, Y) :- 
    output(Goal, _, _, _),  
    findall(Predicate, rule(_ implies Goal/Predicate), GoalPredicates),
    maplist(capped_degree(Goal, X), GoalPredicates, Ys),
    max_list(Ys, Y).

capped_degree(Input, X, Predicate, Y) :- 
    degree(Input/Predicate, X, Y0),
    max_satisfaction(Input/Predicate, M),
    Y is min(Y0, M).

show_curves :- 
    findall(definition(Input, Predicates, Left, Right), definition(Input, Predicates, Left, Right), Definitions),
    writeln(Definitions),
    maplist(show_curvers_for_def, Definitions).

show_curvers_for_def(definition(Input, Predicates, Left, Right)) :-
    maplist(show_curvers_for_pred(Input, Left, Right), Predicates).

show_curvers_for_pred(Input, Left, Right, Predicate) :-
    range(Left, 1, Right, Range),
    maplist(degree(Input/Predicate), Range, Results),
    write("Cuvrve for "), write(Input/Predicate), write(": "), writeln(Results).

eval(rule(Antecedents implies Consequent), Result) :- 
    eval_consequent(rule(Antecedents implies Consequent), ConsequentResult),
    assertz(max_satisfaction(Consequent, ConsequentResult)),
    Result = [Consequent, ConsequentResult],
    write("Rule: "), write(Antecedents implies Consequent), write(" Degree: "), writeln(ConsequentResult), nl.

eval_consequent(rule(A implies _), Result) :- eval_antecedent(A, Result).

eval_consequent(rule(A or B implies _), Result) :-
    eval_antecedent(A, ResultA),
    eval_antecedent(B, ResultB),
    Result is max(ResultA, ResultB).

eval_consequent(rule(A and B implies _), Result) :- 
    eval_antecedent(A, ResultA),
    eval_antecedent(B, ResultB),
    Result is min(ResultA, ResultB).

eval_antecedent(Input/Predicate, Result) :-
    rating(Input, Rating),
    degree(Input/Predicate, Rating, Result),
    write("Premise: "), write(Input/Predicate), write(" Degree: "), write(Result), nl.

ask_rating(Input) :-
    write(Input), write(" > "),
    read_line_to_string(user_input, RatingString),
    (RatingString = "stop" -> halt; true),
    number_string(RatingNumber, RatingString),
    assertz(rating(Input, RatingNumber)).

lin_coef(dec, X1, X2, A, B) :- A is 1 / (X1 - X2), B is - X2 / (X1 - X2).
lin_coef(inc, X1, X2, A, B) :- A is 1 / (X2 - X1), B is - X1 / (X2 - X1).

lin(Dir, Left, Right, X, Y) :- 
    Left =< X, X =< Right,
    lin_coef(Dir, Left, Right, A, B),
    Y is A * X + B.

:- dynamic inputs/1.
:- dynamic output/1.
:- dynamic degree/3.
:- dynamic rule/1.
:- dynamic definition/3.
load_kb(Filename) :-
	see(Filename),
    repeat,
        read(Term),
        ( Term == end_of_file; assertz(Term), fail),
	seen.