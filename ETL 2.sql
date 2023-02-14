USE [Thesis Data]

SELECT Fah_exp.["hhnum"], sum(Fah_exp.["fah_exp_total"])as 'total_fah_exp', COUNT(["fah_exp_total"]) as 'num_meals_fah' 
FROM Fah_exp
Right JOIN dbo.Household_DataSO ON dbo.Household_DataSO.household_num = Fah_exp.["hhnum"] 
WHERE Fah_exp.["fah_exp_total"] > 0
GROUP BY ["hhnum"]
ORDER BY ["hhnum"]

SELECT fafh_exp.["hhnum"], sum(fafh_exp.["fafh_exp_total"]) as 'total_fafh_exp', COUNT(["fafh_exp_total"]) as 'num_meals_fafh' 
FROM fafh_exp
Right JOIN dbo.Household_DataSO ON dbo.Household_DataSO.household_num = Fafh_exp.["hhnum"] 
WHERE Fafh_exp.["fafh_exp_total"] > 0
GROUP BY ["hhnum"]
ORDER BY ["hhnum"]

DELETE FROM FAH_totals WHERE num_meals_fah = 0;
DELETE FROM FAFH_totals WHERE num_meals_fafh = 0;

SELECT *
FROM FAH_totals
ORDER BY ["hhnum"]

SELECT *
FROM FAFH_totals
ORDER BY ["hhnum"]

SELECT DISTINCT FAH_totals.["hhnum"], FAH_totals.total_fah_exp, FAFH_totals.total_fafh_exp, FAH_totals.num_meals_fah, FAFH_totals.num_meals_fafh
FROM FAH_totals
INNER JOIN dbo.FAFH_totals on dbo.FAFH_totals.["hhnum"] = FAH_totals.["hhnum"]
ORDER BY FAH_totals.["hhnum"]