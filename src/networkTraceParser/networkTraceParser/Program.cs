// See https://aka.ms/new-console-template for more information

using System.Globalization;
using CsvHelper;
using Microsoft.VisualBasic.FileIO;
using networkTraceParser.Models;

Console.WriteLine("Hello, World!");
using var reader = new StreamReader("../../../digital.csv");
using var csv = new CsvReader(reader,  CultureInfo.InvariantCulture);
csv.Context.RegisterClassMap<DataInstanceMap>();
var records = csv.GetRecords<DataInstance>().ToList();
Console.WriteLine(records.Count);
var bits = (
    from record in records 
    where records.IndexOf(record) != 0 
    let priorRecord = records[records.IndexOf(record) - 1] 
    where priorRecord.Channel3 == 0 && record.Channel3 == 1 
    select new Bit(priorRecord, record))
    .ToList();

Console.WriteLine(bits.Count);