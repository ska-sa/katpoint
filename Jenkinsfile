#!groovy

@Library('katsdpjenkins') _
katsdp.killOldJobs()
katsdp.standardBuild(python3: true)
katsdp.mail('ludwig@ska.ac.za')
