resolution(KB) :- member([], KB), !.
resolution(KB) :- 
    member(L1, KB),
    member(L2, KB),
    resolve(L, L1, L2),
    \+ member(L, KB),
    resolution([L | KB]), !.

resolve(L, L1, L2) :- resolve_internal(L, L1, L2).
resolve(L, L1, L2) :- resolve_internal(L, L2, L1).
resolve_internal(L, L1, L2) :- 
    select(R, L1, L1_prime), 
    select(not(R), L2, L2_prime),
    append(L1_prime, L2_prime, L_prime),
    list_to_set(L_prime, L).

negate(not(A), A) :- !.
negate(A, not(A)).

kb_test([
	[f(X), X],
     [not(f(a)), a],
     [not(a)]
   ]).

kb_test1([
	[a, b],
           [not(a)],
           [not(b)]
         ]).

kb_assignment([
     [guilty(X), not(brokeArt193(X))],
     [guilty(X), not(brokeArt194(X))],
     [brokeArt193(X), not(battery(X, h(X))), isLongerThan90Days(h(X))],
     [brokeArt194(X), not(battery(X, h(X))), not(isLongerThan90Days(h(X)))],
     [battery(X, h(X)), 
     	not(violence(v(X), X, y(X))), 
     	not(hospitalization(h(X), y(X))), 
     	not(causality(v(X), h(X)))
     ],
     [violence(v(ion), ion, y(ion))],
     [hospitalization(h(ion), y(ion))],
     [isLongerThan90Days(h(ion))],
     [causality(v(ion), h(ion))],
     [not(guilty(ion))]
              ]).

