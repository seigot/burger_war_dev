#!/bin/bash -eu
# auto install keyboard configuration

# 　1. Afghani                   34. Faroese                                    67. Moldavian
#   2. Albanian                  35. Filipino                                   68. Mongolian
#   3. Amharic                   36. Finnish                                    69. Montenegrin
#   4. Arabic                    37. French                                     70. Nepali
#   5. Arabic (Morocco)          38. French (Canada)                            71. Norwegian
#   6. Arabic (Syria)            39. French (Democratic Republic of the Congo)  72. Persian
#   7. Armenian                  40. French (Guinea)                            73. Polish
#   8. Azerbaijani               41. French (Togo)                              74. Portuguese
#   9. Bambara                   42. Georgian                                   75. Portuguese (Brazil)
#   10. Bangla                   43. German                                     76. Romanian
#   11. Belarusian               44. German (Austria)                           77. Russian
#   12. Belgian                  45. Greek                                      78. Serbian
#   13. Berber (Algeria, Latin)  46. Hebrew                                     79. Sinhala (phonetic)
#   14. Bosnian                  47. Hungarian                                  80. Slovak
#   15. Braille                  48. Icelandic                                  81. Slovenian
#   16. Bulgarian                49. Indian                                     82. Spanish
#   17. Burmese                  50. Indonesian (Jawi)                          83. Spanish (Latin American)
#   18. Chinese                  51. Iraqi                                      84. Swahili (Kenya)
#   19. Croatian                 52. Irish                                      85. Swahili (Tanzania)
#   20. Czech                    53. Italian                                    86. Swedish
#   21. Danish                   54. Japanese                                   87. Switzerland
#   22. Dhivehi                  55. Japanese (PC-98)                           88. Taiwanese
#   23. Dutch                    56. Kazakh                                     89. Tajik
#   24. Dzongkha                 57. Khmer (Cambodia)                           90. Thai
#   25. English (Australian)     58. Korean                                     91. Tswana
#   26. English (Cameroon)       59. Kyrgyz                                     92. Turkish
#   27. English (Ghana)          60. Lao                                        93. Turkmen
#   28. English (Nigeria)        61. Latvian                                    94. Ukrainian
#   29. English (South Africa)   62. Lithuanian                                 95. Urdu (Pakistan)
#   30. English (UK)             63. Macedonian                                 96. Uzbek
#   31. English (US)             64. Malay (Jawi, Arabic Keyboard)              97. Vietnamese
#   32. Esperanto                65. Maltese                                    98. Wolof
# Country of origin for the keyboard: 
COUNTRY=54

#   1. Japanese 
#   2. Japanese - Japanese (Dvorak)  
#   3. Japanese - Japanese (Kana 86)  
#   4. Japanese - Japanese (Kana)  
#   5. Japanese - Japanese (Macintosh)  
#   6. Japanese - Japanese (OADG 109A)
# Keyboard layout: 
LAYOUT=1

# install expect 
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
	expect

# install keyboard-configuration
printf "
set timeout -1
spawn apt-get install -y keyboard-configuration
expect {
    \"\\\\\\\\\[More\\\\\\\\\]\" { send \"\\\n\"; exp_continue; }
    \"Country of origin for the keyboard: \" { send \"${COUNTRY}\\\n\"; }
}
expect {
    \"\\\\\\\\\[More\\\\\\\\\]\" { send \"\\\n\"; exp_continue; }
    \"Keyboard layout: \" { send \"${LAYOUT}\\\n\"; }
}
expect eof
" | expect
