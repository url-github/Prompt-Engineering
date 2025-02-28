def sliding_window(text, window_size, step_size): 
  if window_size > len(text) or step_size < 1:
    return []
  return [text[i:i+window_size] for i in range(0, len(text) - window_size + 1, step_size)]
text = "To jest przykładu podziału z użyciem okna przesuwnego."
window_size = 20
step_size = 5
chunks = sliding_window(text, window_size, step_size) 
for idx, chunk in enumerate(chunks):
    print(f"Fragment {idx + 1}: {chunk}")
