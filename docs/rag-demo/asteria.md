# Asteria Context Kernel

Asteria stores the complete conversation in ContextMemory. ContextBuilder creates a budgeted PreparedContext before each model request. Old turns can be summarized while the current turn remains complete.

The AgentLoop separates a user Turn from individual model Steps. A Step may call tools, receive Tool Results, and then call the model again.
