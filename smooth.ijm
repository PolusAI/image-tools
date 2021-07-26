#@ String input
#@ String output
id = getImageID(input)
print(id)
select(id);
run("Smooth");
saveAs(output);
close(input);
close(output);