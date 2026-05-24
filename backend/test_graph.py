from graph import app_graph

# Test normal query
result = app_graph.invoke({
    "question": "What is hostel fee?"
})

print(result)

# Test escalation query
result2 = app_graph.invoke({
    "question": "I want to talk to HOD"
})

print(result2)