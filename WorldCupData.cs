using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Api;

namespace WorldCup
{
    public class WorldCupData
    {
        [Column("0")]
        public float Year;

        [Column("2")]
        public string Stage;

        [Column("5")]
        public string HomeTeam;

        [Column("6")]
        public float HomeTeamGoals;

        [Column("8")]
        public string AwayTeam;

        [Column("10")]
        public float Attendance;

        [Column("13")]
        public string Referee;
    }

    public class WorldCupPrediction
    {
        [ColumnName("Score")]
        public float HomeTeamGoals;
    }
}
