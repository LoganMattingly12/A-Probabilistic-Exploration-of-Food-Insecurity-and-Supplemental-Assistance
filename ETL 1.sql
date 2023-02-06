
--02/03/2023--

-- data type evaluation
SELECT 
TABLE_CATALOG,
TABLE_SCHEMA,
TABLE_NAME, 
COLUMN_NAME, 
DATA_TYPE 
FROM INFORMATION_SCHEMA.COLUMNS
where TABLE_NAME = 'Household_Data'

-- create new table for demographic data
CREATE TABLE Demographics( 
	Household_Number INT,
	Household_Size INT,
	Region INT,
	Rural INT,
	Income FLOAT,
	Target_Group INT,
	SNAP INT,
	Food_Sufficient INT
);

-- fill new table with data from main table only if snap program was used
INSERT INTO Demographics (Household_Number,Household_Size,Region,Rural,Income,Target_Group,SNAP,Food_Sufficient)
SELECT household_num, household_size, region, rural, Income_avg_mon, target_group, snap_ever, food_sufficient_score
FROM Household_Data
WHERE snap_ever = 1

-- test new table
SELECT *
FROM Demographics

--Run test join inorder to prevent errors
SELECT Demographics.Household_Number, Clusters.Household
FROM  dbo.Demographics
INNER JOIN Clusters ON Demographics.Household_Number = Clusters.Household

-- create new table for detrmined clusters found in R code
CREATE TABLE Demographics2 ( 
	Household_Number INT,
	Household_Size INT,
	Region INT,
	Rural INT,
	Income FLOAT,
	Target_Group INT,
	SNAP INT,
	Food_Sufficient INT,
	Cluster INT
);

--Actual data merege
INSERT INTO Demographics2 (Household_Number,Household_Size,Region,Rural,Income,Target_Group,SNAP,Food_Sufficient,Cluster)
SELECT DISTINCT Demographics.Household_Number, Demographics.Household_Size, Demographics.Region, Demographics.Rural, Demographics.Income,Demographics.Target_Group, Demographics.SNAP, Demographics.Food_Sufficient,Clusters.Cluster
FROM  dbo.Demographics
INNER JOIN Clusters ON Demographics.Household_Number = Clusters.Household

-- Insure no duplicates are within the new table
SELECT Count(Household_Number) AS DuplicateRanks
FROM Demographics2
GROUP BY Household_Number
HAVING COUNT(Household_Number)>1;

--Test new table
SELECT *
FROM Demographics2
ORDER BY Household_Number

--02/05/2023--

--copy main foodaps data conditional on snap benifits
SELECT * INTO Household_Data_SO
FROM Household_Data
WHERE snap_ever = 1
ORDER BY household_num

--Test join
ALTER TABLE Household_Data_SO
ADD CLuster INT

--Add clusters to snap table
SELECT * INTO Household_DataSO
FROM Household_Data_SO
LEFT JOIN Clusters2
ON Household_Data_SO.household_num = Clusters2.labels1

-- Remove dummy table
DROP TABLE Household_Data_SO

-- Remove unwanted column
ALTER TABLE Household_DataSO
DROP COLUMN labels1

-- Test new table
SELECT household_num, Income_avg_mon, household_poverty_guideline
FROM Household_DataSO
WHERE Cluster = 2

