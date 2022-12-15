#                                             Identifying Must Play Games:
                                           
  In this project I will be looking at video game data I pulled from kaggle to learn more about what aspects of a game have the most impact on it chance to be categorized as a 'must play game' which is a standard defined by metacritic as a game that has achieved a metacritic rating of and over 90%. These ratings are achieved by taking wighted averages of ratings where the weight is based upon the 'quality' and 'stature' of the critic . It will be useful to  first identify features that are relevant to a games 'must play' status by exploring the data and running statistical tests on different attributes of my dataset. I will than use these relevant features in order to create several models that I can modify until I find the parameters that most accurately predict a games 'must play' status.
                                           
#                                               Pipeline:
Plan: My plan is to create models that will accurately predict 'must play' games by using the features in my videogame dataset I have pulled from kaggle.
                                            
Aquire: acquire data through functions saved in my acquire.py along with my encrypted credentials.

Prepare: Prepare and clean my data in such a way that I can plug it in to my classification models without causing error and with the prepared data still holding                true to its original meaning. This includes:
        - dropping all rows with nulls for the column metacritic as this is where I am deriving my target.
        - encoding all categorical variables.
        - scaling my numeric variables to numbers between 0 and 1.
        - dropping unnecessary columns. Many Included in the dataset would have information that I would not be privy to upon a games release.
        - dropping all remaining nulls as this is only about 100 more rows.

Explore: Ask statistical questions of my data and create visualizations of the results in order to improve comprehension by people reading the report without clarifications from my presentation.

Model:Create models that try to accurately predict 'good game' status, along with visuals that highlight important findings and help the audiance understand their significance. Validate that all of your models are accurate not just on your train data but on outside data as well.

Deliver: Put all of my findings together in a final report where I make things as easy to understand as possible.

#                                          QUESTIONS TO ANSWER:

1.  Are PC games equally likely to be 'must play games' as the general population of platforms?
        Hnull - PC games are equally likely to be good games as the general population.
        Halt - PC games are not equally likely to be good games compared to the population.
2.  Are shooters equally likely to be 'must play games' as the general population of genres?
        Hnull - shooters are equally likely to be 'must play games' as other genres.
        Halt - shooters are not equally likely to be 'must play games' as other genres.
3. Are games published by blizzard equally likely to be 'must play games' as the general population of developers?
        Hnull- blizzard games are equally likely to be 'must play games' as other developers.
        Halt- blizzard games are not equally likely to be 'must play games' as other developers.
4. Are Indie-Games equally likely to be 'must play games' than the general population of genres?
        Hnull - Indie-Games are equally likely to be 'must play games' as other genres.
        Halt - Indie-Games are not equally likely to be 'must play games' as other genres.
#                                              Data Library

<class 'pandas.core.frame.DataFrame'>
Int64Index: 2755 entries, 264546 to 253411
Data columns (total 148 columns):
 #  |  Column                                       | Dtype  
--- |  ------                                       | -----  
 0  |  metacritic                                   | float64
 1  |  playtime                                     | int64  
 2  |  achievements_count                           | int64  
 3  |  game_series_count                            | int64  
 4  |  added_status_yet                             | int64  
 5  |  added_status_toplay                          | int64  
 6  |  metacritic_good_game                         | float64
 7  |  Publisher_Square_Enix                        | object 
 8  |  Publisher_Electronic_Arts                    | object 
 9  |  Publisher_Enhance_Games                      | object 
 10 |  Publisher_Valve                              | object 
 11 |  Publisher_Klei_Entertainment                 | object 
 12 |  Publisher_LucasArts_Entertainment            | object 
 13 |  Publisher_8-4                                | object 
 14 |  Publisher_tobyfox                            | object 
 15 |  Publisher_Aspyr                              | object 
 16 |  Publisher_Bethesda_Softworks                 | object 
 17 |  Publisher_Funcom                             | object 
 18 |  Publisher_Focus_Home_Interactive             | object 
 19 |  Publisher_Warner_Bros._Interactive           | object 
 20 |  Publisher_Activison                          | object 
 21 |  Publisher_Ubisoft_Entertainment              | object 
 22 |  Publisher_Interplay_Productions              | object 
 23 |  Publisher_Matt_Makes_Games                   | object 
 24 |  Publisher_Blizzard_Entertainment             | object 
 25 |  Publisher_Sony_Computer_Entertainment        | object 
 26 |  Publisher_CD_PROJEKT_RED                     | object 
 27 |  Publisher_Rockstar_Games                     | object 
 28 |  Publisher_Rare                               | object 
 29 |  Publisher_Konami                             | object 
 30 |  Publisher_GT_Interactive_Software            | object 
 31 |  Publisher_MacSoft                            | object 
 32 |  Publisher_Annapurna_Games                    | object 
 33 |  Publisher_Bandai_Namco_Entertainment         | object 
 34 |  Publisher_Atlus                              | object 
 35 |  Publisher_Rocketcat                          | object 
 36 |  Publisher_1C-SoftClub                        | object 
 37 |  Publisher_EA_Sports_BIG                      | object 
 38 |  Publisher_Corporation_Polytron               | object 
 39 |  Publisher_FromSoftware                       | object 
 40 |  Publisher_Paradox_Interactive                | object 
 41 |  Publisher_Buka_Entertainment                 | object 
 42 |  Publisher_Sony_Interactive_Entertainment     | object 
 43 |  Publisher_Deep_Silver                        | object 
 44 |  Publisher_Number_None                        | object 
 45 |  Publisher_id_Software                        | object 
 46 |  Publisher_Asmodee_Digital                    | object 
 47 |  Publisher_Nightdive_Studios                  | object 
 48 |  Publisher_Microsoft_Studios                  | object 
 49 |  Publisher_Capcom                             | object 
 50 |  Publisher_2K_Games                           | object 
 51 |  Publisher_Out_of_the_Park_Developments       | object 
 52 |  Publisher_The_Quantum_Astrophysicists_Guild  | object 
 53 |  Publisher_Overhaul_Games                     | object 
 54 |  Publisher_Nintendo                           | object 
 55 |  Publisher_Supergiant_Games                   | object 
 56 |  Publisher_Namco                              | object 
 57 |  Publisher_Infogrames                         | object 
 58 |  Publisher_Activision_Value_Publishing        | object 
 59 |  Publisher_Ukiyo_Publishing                   | object 
 60 |  Publisher_Atari                              | object 
 61 |  Publisher_Polytron                           | object 
 62 |  Publisher_Eidos_Interactive                  | object 
 63 |  Publisher_Active                             | object 
 64 |  Publisher_Square                             | object 
 65 |  Publisher_Days_of_Wonder                     | object 
 66 |  Publisher_Wube_Software                      | object 
 67 |  Publisher_Toby_Fox                           | object 
 68 |  Publisher_Stardock_Entertainment             | object 
 69 |  Publisher_Gameloft                           | object 
 70 |  Publisher_Telltale_Games                     | object 
 71 |  Publisher_Konami_Digital_Entertainment-US    | object 
 72 |  Publisher_Yacht_Club_Games                   | object 
 73 |  Publisher_Trapdoor                           | object 
 74 |  Publisher_Take_Two_Interactive               | object 
 75 |  Publisher_Chillingo                          | object 
 76 |  Publisher_Disney_Interactive                 | object 
 77 |  Publisher_Feral_Interactive                  | object 
 78 |  Publisher_Xbox_Game_Studios                  | object 
 79 |  Publisher_Rocketcat_Games                    | object 
 80 |  Publisher_Hothead_Games                      | object 
 81 |  Publisher_SEGA                               | object 
 82 |  Publisher_Annapurna_Interactive              | object 
 83 |  Publisher_Larian_Studios                     | object 
 84 |  Publisher_BANDAI_NAMCO_Entertainment_America | object 
 85 |  Publisher_Activision_Blizzard                | object 
 86 |  Platform_Xbox_One                            | object 
 87 |  Platform_Nintendo_Switch                     | object 
 88 |  Platform_iOS                                 | object 
 89 |  Platform_Nintendo_3DS                        | object 
 90 |  Platform_PlayStation_3                       | object 
 91 |  Platform_PlayStation_2                       | object 
 92 |  Platform_Wii                                 | object 
 93 |  Platform_Game_Boy_Advance                    | object 
 94 |  Platform_Wii_U                               | object 
 95 |  Platform_Web                                 | object 
 96 |  Platform_GameCube                            | object 
 97 |  Platform_PlayStation                         | object 
 98 |  Platform_Atari_ST                            | object 
 99 |  Platform_SNES                                | object 
 100|  Platform_Game_Boy_Color                      | object 
 101|  Platform_Atari_2600                          | object 
 102|  Platform_Game_Boy                            | object 
 103|  Platform_Apple_II                            | object 
 104|  Platform_Game_Gear                           | object 
 105|  Platform_Xbox_Series_S/X                     | object 
 106|  Platform_SEGA_Saturn                         | object 
 107|  Platform_PC                                  | object 
 108|  Platform_macOS                               | object 
 109|  Platform_Xbox_360                            | object 
 110|  Platform_Xbox                                | object 
 111|  Platform_Classic_Macintosh                   | object 
 112|  Platform_PSP                                 | object 
 113|  Platform_Nintendo_DSi                        | object 
 114|  Platform_Genesis                             | object 
 115|  Platform_Atari_8-bit                         | object 
 116|  Platform_NES                                 | object 
 117|  Platform_PlayStation_5                       | object 
 118|  Platform_PlayStation_4                       | object 
 119|  Platform_Android                             | object 
 120|  Platform_Nintendo_DS                         | object 
 121|  Platform_Commodore_/_Amiga                   | object 
 122|  Platform_Atari_5200                          | object 
 123|  Platform_3DO                                 | object 
 124|  Platform_PS_Vita                             | object 
 125|  Platform_Neo_Geo                             | object 
 126|  Platform_Linux                               | object 
 127|  Platform_Nintendo_64                         | object 
 128|  Platform_Dreamcast                           | object 
 129|  Genre_Adventure                              | bool   
 130|  Genre_Strategy                               | bool   
 131|  Genre_Simulation                             | bool   
 132|  Genre_Casual                                 | bool   
 133|  Genre_Platformer                             | bool   
 134|  Genre_Card                                   | bool   
 135|  Genre_Massively_Multiplayer                  | bool   
 136|  Genre_Educational                            | bool   
 137|  Genre_Family                                 | bool   
 138|  Genre_Action                                 | bool   
 139|  Genre_RPG                                    | bool   
 140|  Genre_Puzzle                                 | bool   
 141|  Genre_Racing                                 | bool   
 142|  Genre_Sports                                 | bool   
 143|  Genre_Indie                                  | bool   
 144|  Genre_Board_Games                            | bool   
 145|  Genre_Fighting                               | bool   
 146|  Genre_Arcade                                 | bool   
 147|  Genre_Shooter                                | bool  

   
#                                                    Key Findings:
 
I found 4 important features to focus on in the name of predicting 'must play games' , and created a few good models that beat the baseline, my best testing at 98.7 % accurate with a precision of 100% on must play games. In the future I would like a little more time to explore and tamper with the features I include in my models as I believe doing so could allow me to capture the 'must play games' my model is not finding.
 
#                                                  How To Reproduce:
In order to reproduce my report you will need:
- the video_game.csv 
- you will need to download and move my wrangle.py, explore.py, and Modeling.py into the same directory
-Than you will need to run my report from the same directory.

