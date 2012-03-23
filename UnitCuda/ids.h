#pragma once

// INLINE DATA STRUCTS

#define DEQUE DEQUE_ROOT
#define DEQUE_ROOT(TYPE, NAME)																			\
TYPE* NAME##_head;																						\
TYPE* NAME##_tail;

#define DEQUE_NODE(TYPE, NAME)																			\
TYPE* NAME##_next;																						\
TYPE* NAME##_prev;

#define DEQUE_HEAD(ROOT, NAME)		((ROOT)->NAME##_head)
#define DEQUE_TAIL(ROOT, NAME)		((ROOT)->NAME##_tail)

#define DEQUE_NEXT(NODE, NAME)		((NODE)->NAME##_next)
#define DEQUE_PREV(NODE, NAME)		((NODE)->NAME##_prev)

#define DEQUE_PUSH	DEQUE_HEAD_PUSH
#define DEQUE_HEAD_PUSH(ROOT, NAME, NODE)																\
{																										\
	DEQUE_PREV(NODE, NAME) = 0;																			\
	DEQUE_NEXT(NODE, NAME) = DEQUE_HEAD(ROOT, NAME);													\
	DEQUE_HEAD(ROOT, NAME) = (NODE);																	\
	if( DEQUE_TAIL(ROOT, NAME) == 0 )																	\
		DEQUE_TAIL(ROOT, NAME) = (NODE);																\
}

#define DEQUE_TAIL_PUSH(ROOT, NAME, NODE)																\
{																										\
	DEQUE_NEXT(NODE, NAME) = 0;																			\
	DEQUE_PREV(NODE, NAME) = DEQUE_TAIL(ROOT, NAME);													\
	DEQUE_TAIL(ROOT, NAME) = (NODE);																	\
	if( DEQUE_HEAD(ROOT, NAME) == 0 )																	\
		DEQUE_HEAD(ROOT, NAME) = (NODE);																\
}

#define DEQUE_POP	DEQUE_HEAD_POP
#define DEQUE_HEAD_POP(ROOT, NAME, RESULT)																\
{																										\
	(RESULT) = DEQUE_HEAD(ROOT, NAME);																	\
	if( DEQUE_HEAD(ROOT, NAME) )																		\
	{																									\
		DEQUE_HEAD(ROOT, NAME) = DEQUE_NEXT(DEQUE_HEAD(ROOT, NAME), NAME);								\
		DEQUE_PREV(DEQUE_HEAD(ROOT, NAME), NAME) = 0;													\
	}																									\
	else																								\
		DEQUE_HEAD(ROOT, NAME) = DEQUE_TAIL(ROOT, NAME) = 0;											\
}

#define DEQUE_TAIL_POP(ROOT, NAME, RESULT)																\
{																										\
	(RESULT) = DEQUE_TAIL(ROOT, NAME);																	\
	if( DEQUE_TAIL(ROOT, NAME) )																		\
	{																									\
		DEQUE_TAIL(ROOT, NAME) = DEQUE_PREV(DEQUE_TAIL(ROOT, NAME), NAME);								\
		DEQUE_NEXT(DEQUE_TAIL(ROOT, NAME), NAME) = 0;													\
	}																									\
	else																								\
		DEQUE_HEAD(ROOT, NAME) = DEQUE_TAIL(ROOT, NAME) = 0;											\
}

#define DEQUE_UNLINK(ROOT, NAME, NODE)																	\
{																										\
	if( DEQUE_NEXT(NODE, NAME) )																		\
		DEQUE_PREV(DEQUE_NEXT(NODE, NAME), NAME) = DEQUE_PREV(NODE, NAME);								\
	if( DEQUE_PREV(NODE, NAME) )																		\
		DEQUE_NEXT(DEQUE_PREV(NODE, NAME), NAME) = DEQUE_NEXT(NODE, NAME);								\
	if( DEQUE_HEAD(ROOT, NAME) == (NODE) )																\
		DEQUE_HEAD(ROOT, NAME) = DEQUE_NEXT(NODE, NAME);												\
	if( DEQUE_TAIL(ROOT, NAME) == (NODE) )																\
		DEQUE_TAIL(ROOT, NAME) = DEQUE_PREV(NODE, NAME);												\
	DEQUE_NEXT(NODE, NAME) = DEQUE_PREV(NODE, NAME) = 0;												\
}

#define DEQUE_NOT_LINKED_IN(ROOT, NAME, NODE)																	\
(																										\
	DEQUE_TAIL(ROOT, NAME) != (NODE) &&																	\
	DEQUE_HEAD(ROOT, NAME) != (NODE) &&																	\
	DEQUE_PREV(NODE, NAME) == 0 &&																		\
	DEQUE_NEXT(NODE, NAME) == 0																			\
)

#define DEQUE_FOREACH DEQUE_HEAD_FOREACH
#define DEQUE_HEAD_FOREACH(ROOT, NAME, ITERATOR)														\
for(auto ITERATOR = DEQUE_HEAD(ROOT, NAME); ITERATOR; ITERATOR = DEQUE_NEXT(ITERATOR, NAME))

#define DEQUE_FOREACH_REVERSE DEQUE_TAIL_FOREACH
#define DEQUE_TAIL_FOREACH(ROOT, NAME, ITERATOR)														\
for(auto ITERATOR = DEQUE_TAIL(ROOT, NAME); ITERATOR; ITERATOR = DEQUE_PREV(ITERATOR, NAME))
