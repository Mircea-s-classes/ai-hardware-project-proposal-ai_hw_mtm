// See https://aka.ms/new-console-template for more information

using System.Globalization;
using CsvHelper;
using networkTraceParser.Models;

Console.WriteLine("Hello, World!");
using var reader = new StreamReader("../../../digital.csv");
using var csv = new CsvReader(reader,  CultureInfo.InvariantCulture);
csv.Context.RegisterClassMap<DataInstanceMap>();
var records = csv.GetRecords<DataInstance>().ToList();
Console.WriteLine(records.Count);
var bits = records
    .Zip(records.Skip(1), (prev, curr) => new { prev, curr }) 
    .Where(pair => pair.prev.Channel3 == 0 && pair.curr.Channel3 == 1)
    .Select(pair => new Bit(pair.prev, pair.curr))
    .ToList();

Console.WriteLine(bits.Count);