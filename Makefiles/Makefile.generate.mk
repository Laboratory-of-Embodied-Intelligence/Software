
generate:
	@echo "$(sep)Automated files generation"
	@echo
	@echo 'Generation of documentation'
	@echo
	@echo '- `make generate-all`:              Generates everything.'
	@echo '- `make generate-help`:             Generates help.'
	@echo '- `make generate-easy_node`:        Generates the easy node documentation.'
	@echo '- `make generate-easy_node-clean`:  Cleans the generated files.'


generate-all: \
	generate-help\
	generate-easy_node

out_help=Makefiles/help.autogenerated.md

generate-help:
	echo > $(out_help)
	echo "\n\n<div id='makefiles-autogenerated' markdown='1'>\n\n" >> $(out_help)
	echo '## Makefile help {#makefiles-help}' >> $(out_help)
	$(MAKE) -s all >> $(out_help)
	echo "\n\n</div>\n\n" >> $(out_help)


generate-easy_node:
	rosrun easy_node generate_docs

generate-easy_node-clean:
	find . -name '*autogenerated.md' -delete