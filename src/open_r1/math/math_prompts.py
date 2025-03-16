import random

def build_conv(question, prefix=""):
    print(f"New Prompts")
    messages = [
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": "<think>\n"
        }
    ]
    if prefix != "":
        messages[1]["content"] = messages[1]["content"] + prefix
    return messages

def sample_prefixes(text, number_of_phrases):
    delimiters = [
        "\nWait",
        "\nBut wait",
        "\nAlternatively",
        "\nIs there another way to think about this?",
        "\nBut let me double-check",
        "\nBut hold on",
        "</think>",
    ]

    # Initialize result with the full text
    segments = [text]

    # For each delimiter
    for delimiter in delimiters:
        # Create a new list for the split segments
        new_segments = []

        # Split each existing segment
        for segment in segments:
            # Split the segment
            split_parts = segment.split(delimiter)

            # Add first part without delimiter
            if split_parts[0]:  # Skip empty strings
                new_segments.append(split_parts[0])

            # Add remaining parts with delimiter
            for part in split_parts[1:]:
                new_segments.append(delimiter + part)

        # Update the segments list
        segments = new_segments
        
    # Remove any empty segments that might have been created
    segments = [segment for segment in segments if segment != ""]
    
    if len(segments) > number_of_phrases:
        segment_indices = sorted(random.sample(range(len(segments)), number_of_phrases))
    else:
        segment_indices = list(range(len(segments)))
    # Process only the sampled steps
    prefixes = []
    segment_indices.sort()
    print(segment_indices)
    for j in segment_indices:
        if j == 0:
            prefix = ""
        else:
            prefix = "".join(segments[:j])
        prefixes.append(prefix)
    return prefixes