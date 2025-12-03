using CsvHelper.Configuration;

namespace networkTraceParser.Models;

public sealed class DataInstanceMap : ClassMap<DataInstance>
{
    public DataInstanceMap()
    {
        Map(m => m.Timestamp).Name("Time [s]");
        Map(m => m.Channel0).Name("Channel 0");
        Map(m => m.Channel1).Name("Channel 1");
        Map(m => m.Channel2).Name("Channel 2");
        Map(m => m.Channel3).Name("Channel 3");
        Map(m => m.Channel4).Name("Channel 4");
        Map(m => m.Channel5).Name("Channel 5");
        Map(m => m.Channel6).Name("Channel 6");
        Map(m => m.Channel7).Name("Channel 7");
        Map(m => m.Channel8).Name("Channel 8");
        Map(m => m.Channel9).Name("Channel 9");
    }
}