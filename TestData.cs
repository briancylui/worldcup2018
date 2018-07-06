using System;
using System.Collections.Generic;
using System.Text;

namespace WorldCup
{
    static class TestData
    {
        internal static readonly WorldCupData TestEngland = new WorldCupData
        {
            Year = 2018f,
            Stage = "Quarter-finals",
            HomeTeam = "England",
            HomeTeamGoals = 0, // predict it
            AwayTeam = "Sweden",
            Attendance = 45963f,
            Referee = "KUIPERS (NED)",
        };

        internal static readonly WorldCupData TestSweden = new WorldCupData
        {
            Year = 2018f,
            Stage = "Quarter-finals",
            HomeTeam = "Sweden",
            HomeTeamGoals = 0, // predict it
            AwayTeam = "England",
            Attendance = 45963f,
            Referee = "KUIPERS (NED)",
        };
    }
}
