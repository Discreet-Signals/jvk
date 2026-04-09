# Strips trailing commas from JSON files (JUCE 8 moduleinfo.json bug)
if (EXISTS "${MODULEINFO}")
    file(READ "${MODULEINFO}" content)
    # Remove trailing comma before } or ]
    string(REGEX REPLACE ",[ \t\r\n]*}" "}" content "${content}")
    string(REGEX REPLACE ",[ \t\r\n]*\\]" "]" content "${content}")
    file(WRITE "${MODULEINFO}" "${content}")
endif()
