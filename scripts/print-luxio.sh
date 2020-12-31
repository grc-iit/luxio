#!/bin/bash
jq 'map_values(.val)' $1
